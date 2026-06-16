"""集成测试 - 覆盖所有模块: parser, context, layer, cache, builder, packager。

运行:
    python -m pytest tests/test_all.py -v
    或  python tests/test_all.py (不依赖 pytest, 使用内置 unittest)
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mini_builder.parser import parse_dockerfile, InstructionType, BuildStep
from mini_builder.context import BuildContext, EmptyContext
from mini_builder.layer import Layer, LayerStore, LayeredFilesystem
from mini_builder.cache import LayerCache
from mini_builder.builder import Builder, BuildState, MiniShell
from mini_builder.packager import Packager


class TestParser(unittest.TestCase):
    """测试 Dockerfile 解析。"""

    def test_basic_instructions(self):
        df = """
        FROM ubuntu:22.04 AS base
        RUN apt-get update && apt-get install -y curl
        COPY --from=builder /app/build /app
        ADD source.tar.gz /data/
        ENV PATH=/usr/bin:$PATH FOO=bar
        LABEL org.example.name=test
        ARG VERSION=1.0
        WORKDIR /app
        USER nobody
        EXPOSE 8080 9090
        VOLUME ["/data", "/logs"]
        CMD ["python", "app.py"]
        ENTRYPOINT /bin/sh -c start.sh
        MAINTAINER test@example.com
        """
        steps = parse_dockerfile(df)
        types = [s.type for s in steps]
        self.assertEqual(types, [
            InstructionType.FROM,
            InstructionType.RUN,
            InstructionType.COPY,
            InstructionType.ADD,
            InstructionType.ENV,
            InstructionType.LABEL,
            InstructionType.ARG,
            InstructionType.WORKDIR,
            InstructionType.USER,
            InstructionType.EXPOSE,
            InstructionType.VOLUME,
            InstructionType.CMD,
            InstructionType.ENTRYPOINT,
            InstructionType.MAINTAINER,
        ])

    def test_from_as(self):
        steps = parse_dockerfile("FROM alpine:3.18 AS builder")
        self.assertEqual(steps[0].args[0], "alpine:3.18")
        self.assertEqual(steps[0].kwargs["as"], "builder")

    def test_continuation_lines(self):
        df = """RUN apt-get update && \
    apt-get install -y \
        curl \
        wget
"""
        steps = parse_dockerfile(df)
        self.assertEqual(len(steps), 1)
        self.assertIn("curl", steps[0].args[0])
        self.assertIn("wget", steps[0].args[0])

    def test_comments_and_blank_lines(self):
        df = """# this is a comment
        # another comment

        FROM scratch

        # trailing comment
        RUN echo hi
        """
        steps = parse_dockerfile(df)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].type, InstructionType.FROM)
        self.assertEqual(steps[1].type, InstructionType.RUN)

    def test_copy_multiple_sources(self):
        steps = parse_dockerfile("COPY a.txt b.txt /app/")
        self.assertEqual(steps[0].args, ["a.txt", "b.txt", "/app/"])

    def test_env_kv_form(self):
        steps = parse_dockerfile('ENV FOO=bar BAZ="qux quux"')
        self.assertEqual(steps[0].kwargs["FOO"], "bar")
        self.assertEqual(steps[0].kwargs["BAZ"], "qux quux")

    def test_env_legacy_form(self):
        steps = parse_dockerfile("ENV FOO bar baz qux")
        self.assertEqual(steps[0].kwargs["FOO"], "bar baz qux")


class TestContext(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mib-test-ctx-")
        # 创建一些文件
        (Path(self.tmp) / "a.txt").write_text("hello")
        (Path(self.tmp) / "sub").mkdir()
        (Path(self.tmp) / "sub" / "b.txt").write_text("world")
        (Path(self.tmp) / "sub" / "c.bin").write_bytes(b"\x00\x01\x02")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_resolve_and_safety(self):
        ctx = BuildContext(self.tmp)
        self.assertEqual(ctx.resolve("a.txt"), (Path(self.tmp) / "a.txt").resolve())
        with self.assertRaises(ValueError):
            ctx.resolve("../etc/passwd")

    def test_file_hash(self):
        ctx = BuildContext(self.tmp)
        h = ctx.file_hash("a.txt")
        self.assertEqual(len(h), 64)  # sha256 长度
        self.assertNotEqual(h, "")

    def test_manifest(self):
        ctx = BuildContext(self.tmp)
        self.assertIn("a.txt", ctx.manifest)
        self.assertIn("sub/b.txt", ctx.manifest)
        self.assertIn("sub/c.bin", ctx.manifest)

    def test_content_hash_deterministic(self):
        ctx = BuildContext(self.tmp)
        h1 = ctx.content_hash_for(["a.txt", "sub/"])
        h2 = ctx.content_hash_for(["a.txt", "sub/"])
        self.assertEqual(h1, h2)

    def test_content_hash_changes(self):
        ctx = BuildContext(self.tmp)
        h1 = ctx.content_hash_for(["a.txt"])
        (Path(self.tmp) / "a.txt").write_text("changed")
        ctx2 = BuildContext(self.tmp)
        h2 = ctx2.content_hash_for(["a.txt"])
        self.assertNotEqual(h1, h2)


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mib-test-layer-")
        self.store = LayerStore(os.path.join(self.tmp, "store"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_create_and_store_layer(self):
        layer = self.store.create_layer(parent=None)
        layer.add_string_content("etc/hosts", "127.0.0.1 localhost\n")
        layer.add_string_content("app/data.txt", "hello\n")
        self.store.store_layer(layer)

        loaded = self.store.get_layer(layer.layer_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.layer_id, layer.layer_id)
        self.assertIsNone(loaded.parent_id)

    def test_whiteout_remove(self):
        # 底层: 创建文件
        base = self.store.create_layer(parent=None)
        base.add_string_content("app/foo.txt", "foo")
        base.add_string_content("app/bar.txt", "bar")
        self.store.store_layer(base)

        # 上层: 删除 foo
        upper = self.store.create_layer(parent=base)
        upper.remove_path("app/foo.txt")
        upper.add_string_content("app/baz.txt", "baz")
        self.store.store_layer(upper)

        # 叠加视图
        lfs = LayeredFilesystem([base, upper])
        self.assertIsNotNone(lfs.resolve_path("app/bar.txt")[0])
        self.assertIsNotNone(lfs.resolve_path("app/baz.txt")[0])
        self.assertIsNone(lfs.resolve_path("app/foo.txt")[0], "whiteout 应使文件不存在")

    def test_opaque_marker(self):
        base = self.store.create_layer(parent=None)
        base.add_string_content("var/log/a.log", "old log")
        base.add_string_content("var/log/b.log", "old log")
        self.store.store_layer(base)

        upper = self.store.create_layer(parent=base)
        upper.mark_opaque("var/log")
        upper.add_string_content("var/log/c.log", "new log")
        self.store.store_layer(upper)

        lfs = LayeredFilesystem([base, upper])
        self.assertIsNone(lfs.resolve_path("var/log/a.log")[0], "opaque 应屏蔽下层目录内容")
        self.assertIsNotNone(lfs.resolve_path("var/log/c.log")[0])

    def test_materialize(self):
        base = self.store.create_layer(parent=None)
        base.add_string_content("a.txt", "hello")
        self.store.store_layer(base)

        upper = self.store.create_layer(parent=base)
        upper.add_string_content("b.txt", "world")
        upper.remove_path("a.txt")
        self.store.store_layer(upper)

        out_dir = tempfile.mkdtemp()
        try:
            lfs = LayeredFilesystem([base, upper])
            lfs.materialize(out_dir)
            self.assertFalse((Path(out_dir) / "a.txt").exists())
            self.assertTrue((Path(out_dir) / "b.txt").exists())
            self.assertEqual((Path(out_dir) / "b.txt").read_text(), "world")
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_content_hash_deterministic(self):
        l1 = self.store.create_layer(parent=None)
        l1.add_string_content("x.txt", "data")
        self.store.store_layer(l1)

        l2 = self.store.create_layer(parent=None)
        l2.add_string_content("x.txt", "data")
        self.store.store_layer(l2)

        self.assertEqual(l1.compute_content_hash(), l2.compute_content_hash())

    def test_pack_tar(self):
        layer = self.store.create_layer(parent=None)
        layer.add_string_content("test.txt", "test content\n")
        tar_path = os.path.join(self.tmp, "layer.tar.gz")
        digest, diff_id, size = layer.pack_to_tar(tar_path)
        self.assertTrue(digest.startswith("sha256:"))
        self.assertTrue(diff_id.startswith("sha256:"))
        self.assertGreater(size, 0)
        self.assertTrue(os.path.exists(tar_path))


class TestCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mib-test-cache-")
        self.store = LayerStore(os.path.join(self.tmp, "store"))
        self.cache = LayerCache(os.path.join(self.tmp, "cache"), self.store)
        self.ctx = EmptyContext()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cache_hit_miss(self):
        step = BuildStep(
            type=InstructionType.RUN, raw='RUN echo "hi"', line_no=1, args=['echo "hi"'], flags=["shell"]
        )
        # 首次: miss
        result = self.cache.get_layer_if_cached("", step, self.ctx)
        self.assertIsNone(result)

        # 写入
        layer = self.store.create_layer(parent=None)
        layer.add_string_content("x.txt", "hi")
        self.store.store_layer(layer)
        next_chain = self.cache.record("", step, self.ctx, layer)

        # 再次查找: hit
        result = self.cache.get_layer_if_cached("", step, self.ctx)
        self.assertIsNotNone(result)
        hit_layer, ch = result
        self.assertEqual(hit_layer.layer_id, layer.layer_id)
        self.assertEqual(ch, next_chain)

    def test_chain_propagation(self):
        # 验证 chain_hash 变化会导致缓存失效
        s1 = BuildStep(type=InstructionType.ENV, raw="ENV A=1", line_no=1, kwargs={"A": "1"})
        s2 = BuildStep(type=InstructionType.ENV, raw="ENV B=2", line_no=2, kwargs={"B": "2"})

        # S1 在空 chain 下写入
        l1 = self.store.create_layer(parent=None)
        self.store.store_layer(l1)
        ch1 = self.cache.record("", s1, self.ctx, l1)

        # S2 在 ch1 下写入
        l2 = self.store.create_layer(parent=l1)
        self.store.store_layer(l2)
        ch2 = self.cache.record(ch1, s2, self.ctx, l2)

        # S1 变更: 使用空 chain, 但指令修改
        s1b = BuildStep(type=InstructionType.ENV, raw="ENV A=999", line_no=1, kwargs={"A": "999"})
        result = self.cache.get_layer_if_cached("", s1b, self.ctx)
        self.assertIsNone(result, "修改的指令不应命中旧缓存")

        # S2 在新的 ch1b 下也不会命中
        l1b = self.store.create_layer(parent=None)
        self.store.store_layer(l1b)
        ch1b = self.cache.record("", s1b, self.ctx, l1b)
        result = self.cache.get_layer_if_cached(ch1b, s2, self.ctx)
        self.assertIsNone(result, "父 chain 变化, 后续步骤缓存必须失效")


class TestMiniShell(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="mib-test-shell-")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_echo_mkdir_touch_rm(self):
        sh = MiniShell(self.root, env={}, workdir="/")
        rc = sh.run("mkdir -p /app && touch /app/foo.txt && echo hello world > /app/out.txt")
        self.assertEqual(rc, 0)
        out = Path(self.root) / "app" / "out.txt"
        self.assertTrue(out.exists())
        self.assertIn("hello world", out.read_text())

        rc = sh.run("rm -rf /app")
        self.assertEqual(rc, 0)
        self.assertFalse(out.exists())

    def test_cp_and_cat(self):
        sh = MiniShell(self.root, env={}, workdir="/")
        sh.run("mkdir -p /src && echo data1 > /src/a.txt && echo data2 > /src/b.txt")
        sh.run("mkdir -p /dst && cp /src/a.txt /src/b.txt /dst/")
        a = Path(self.root) / "dst" / "a.txt"
        b = Path(self.root) / "dst" / "b.txt"
        self.assertTrue(a.exists())
        self.assertTrue(b.exists())

    def test_cd_and_pwd(self):
        sh = MiniShell(self.root, env={}, workdir="/")
        sh.run("mkdir -p /a/b/c && cd /a/b && pwd")
        self.assertEqual(sh.cwd, "/a/b")

    def test_env_expansion(self):
        sh = MiniShell(self.root, env={"FOO": "bar"}, workdir="/")
        rc = sh.run("echo $FOO > /out.txt && echo ${FOO:-default} >> /out.txt")
        self.assertEqual(rc, 0)
        lines = (Path(self.root) / "out.txt").read_text().splitlines()
        self.assertEqual(lines[0].strip(), "bar")
        self.assertEqual(lines[1].strip(), "bar")

    def test_and_or_logic(self):
        sh = MiniShell(self.root, env={}, workdir="/")
        # 正确情况: && 后执行
        rc = sh.run("echo first > /out.txt && echo second >> /out.txt")
        self.assertEqual(rc, 0)
        lines = (Path(self.root) / "out.txt").read_text().splitlines()
        self.assertEqual(len(lines), 2)
        # 错误情况: && 后不执行, || 后执行
        sh = MiniShell(self.root, env={}, workdir="/")
        rc = sh.run("false-command || echo fallback > /out2.txt")
        self.assertEqual(rc, 0)
        self.assertTrue((Path(self.root) / "out2.txt").exists())


class TestBuilder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mib-test-builder-")
        self.store = LayerStore(os.path.join(self.tmp, "store"))
        self.cache = LayerCache(os.path.join(self.tmp, "cache"), self.store)
        # 构建上下文: 一个包含文件的目录
        self.ctx_dir = os.path.join(self.tmp, "ctx")
        os.makedirs(self.ctx_dir)
        (Path(self.ctx_dir) / "src.txt").write_text("source content\n")
        (Path(self.ctx_dir) / "sub").mkdir()
        (Path(self.ctx_dir) / "sub" / "nested.txt").write_text("nested\n")
        self.ctx = BuildContext(self.ctx_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_basic_build(self):
        df = """
FROM scratch
WORKDIR /app
ENV MODE=prod
COPY src.txt /app/s.txt
COPY sub/ /app/sub/
RUN echo "build $(echo date)" > /app/build.log && \
    mkdir -p /app/tmp && \
    echo hi > /app/tmp/hi.txt
RUN rm /app/tmp/hi.txt
LABEL version=1.0
EXPOSE 8080
CMD ["/bin/sh", "-c", "echo hi"]
"""
        b = Builder(layer_store=self.store, cache=self.cache, context=self.ctx)
        result = b.build_from_dockerfile(df)
        self.assertTrue(result.success, result.error_message)
        self.assertGreaterEqual(len(result.layers), 5)

        # 验证最终状态
        self.assertEqual(result.final_state.env.get("MODE"), "prod")
        self.assertEqual(result.final_state.workdir, "/app")
        self.assertIn("8080", result.final_state.exposed_ports)
        self.assertEqual(result.final_state.cmd, ["/bin/sh", "-c", "echo hi"])

        # 验证文件: materialize 并检查
        lfs = LayeredFilesystem(result.layers)
        out_dir = os.path.join(self.tmp, "mat")
        lfs.materialize(out_dir)
        s_txt = Path(out_dir) / "app" / "s.txt"
        self.assertTrue(s_txt.exists())
        self.assertIn("source content", s_txt.read_text())

        nested = Path(out_dir) / "app" / "sub" / "nested.txt"
        self.assertTrue(nested.exists())

        build_log = Path(out_dir) / "app" / "build.log"
        self.assertTrue(build_log.exists(), "RUN 指令产物必须存在")

        hi_txt = Path(out_dir) / "app" / "tmp" / "hi.txt"
        self.assertFalse(hi_txt.exists(), "被 RUN rm 删除的文件不应存在")

    def test_cache_reuse(self):
        df = """
FROM scratch
ENV A=1
ENV B=2
ENV C=3
"""
        b1 = Builder(layer_store=self.store, cache=self.cache, context=self.ctx)
        r1 = b1.build_from_dockerfile(df)
        self.assertTrue(r1.success)
        cache_hits_first = sum(1 for e in r1.events if e.kind == "CACHE")
        self.assertEqual(cache_hits_first, 0, "第一次构建不应有缓存命中")

        # 第二次完全相同的构建
        b2 = Builder(layer_store=self.store, cache=self.cache, context=self.ctx)
        r2 = b2.build_from_dockerfile(df)
        self.assertTrue(r2.success)
        cache_hits = sum(1 for e in r2.events if e.kind == "CACHE")
        self.assertGreaterEqual(cache_hits, 3, "第二次构建应全部命中缓存 (ENV*3)")

        # 变更中间一条指令 → 该步骤及其后应失效
        df_modified = """
FROM scratch
ENV A=1
ENV B=999
ENV C=3
"""
        b3 = Builder(layer_store=self.store, cache=self.cache, context=self.ctx)
        r3 = b3.build_from_dockerfile(df_modified)
        self.assertTrue(r3.success)
        build_count = sum(1 for e in r3.events if e.kind == "BUILD")
        self.assertGreaterEqual(build_count, 2, "修改 ENV B 后, ENV B 和 ENV C 都应重新构建")


class TestPackager(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mib-test-pkg-")
        self.store = LayerStore(os.path.join(self.tmp, "store"))
        self.cache = LayerCache(os.path.join(self.tmp, "cache"), self.store)
        self.ctx_dir = os.path.join(self.tmp, "ctx")
        os.makedirs(self.ctx_dir)
        (Path(self.ctx_dir) / "f.txt").write_text("content\n")
        self.ctx = BuildContext(self.ctx_dir)

        # 先做一次简单构建
        df = """
FROM scratch
COPY f.txt /data/f.txt
RUN echo ok > /data/ok.txt
ENV FOO=bar
WORKDIR /data
CMD ["/bin/sh"]
"""
        builder = Builder(layer_store=self.store, cache=self.cache, context=self.ctx)
        self.result = builder.build_from_dockerfile(df)
        self.assertTrue(self.result.success, self.result.error_message)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_oci_layout(self):
        pkg = Packager(scratch_dir=os.path.join(self.tmp, "scratch"))
        out_dir = os.path.join(self.tmp, "oci-out")
        pr = pkg.pack_oci_layout(self.result, out_dir, name="test", tag="v1")
        self.assertEqual(pr.format, "oci")
        # 检查 OCI 目录结构
        oci_root = Path(out_dir)
        self.assertTrue((oci_root / "oci-layout").exists())
        self.assertTrue((oci_root / "index.json").exists())
        self.assertTrue((oci_root / "blobs" / "sha256").is_dir())

        index = json.loads((oci_root / "index.json").read_text())
        self.assertEqual(index["schemaVersion"], 2)
        manifest_digest = index["manifests"][0]["digest"].split(":", 1)[1]
        manifest_file = oci_root / "blobs" / "sha256" / manifest_digest
        self.assertTrue(manifest_file.exists())
        manifest = json.loads(manifest_file.read_text())
        config_digest = manifest["config"]["digest"].split(":", 1)[1]
        config_file = oci_root / "blobs" / "sha256" / config_digest
        cfg = json.loads(config_file.read_text())
        self.assertEqual(cfg["rootfs"]["type"], "layers")
        self.assertGreater(len(cfg["rootfs"]["diff_ids"]), 0)
        self.assertIn("FOO=bar", [e for e in cfg["config"].get("Env", [])])
        self.assertEqual(cfg["config"]["WorkingDir"], "/data")

    def test_docker_tar(self):
        pkg = Packager(scratch_dir=os.path.join(self.tmp, "scratch"))
        tar_path = os.path.join(self.tmp, "image.tar")
        pr = pkg.pack_docker_tar(self.result, tar_path, name="myimg", tag="v1")
        self.assertEqual(pr.format, "docker-tar")
        self.assertTrue(os.path.exists(tar_path))
        # 检查 tar 内容
        with tarfile.open(tar_path, "r") as tf:
            names = tf.getnames()
            manifest_files = [n for n in names if os.path.basename(n) == "manifest.json"]
            self.assertTrue(manifest_files)
            with tf.extractfile(manifest_files[0]) as f:
                mf = json.loads(f.read())
            self.assertIn("myimg:v1", mf[0]["RepoTags"])
            self.assertGreater(len(mf[0]["Layers"]), 0)

    def test_flat_tar(self):
        pkg = Packager(scratch_dir=os.path.join(self.tmp, "scratch"))
        tar_path = os.path.join(self.tmp, "flat.tar.gz")
        pr = pkg.pack_flat_tar(self.result, tar_path)
        self.assertEqual(pr.format, "flat-tar")
        # 解压并检查内容
        extract = os.path.join(self.tmp, "extract")
        os.makedirs(extract)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(extract)
        self.assertTrue((Path(extract) / "data" / "f.txt").exists())
        self.assertTrue((Path(extract) / "data" / "ok.txt").exists())


# ---------------------------------------------------------------------------
# 集成测试: 端到端 demo
# ---------------------------------------------------------------------------


class TestEndToEnd(unittest.TestCase):
    """端到端: 使用 fixtures 中的真实 Dockerfile.demo。"""

    def test_fixture_demo(self):
        root = Path(__file__).parent / "fixtures"
        demo_dockerfile = str(root / "Dockerfile.demo")
        demo_context = str(root / "context")

        ws = tempfile.mkdtemp(prefix="mib-test-e2e-")
        try:
            store = LayerStore(os.path.join(ws, "store"))
            cache = LayerCache(os.path.join(ws, "cache"), store)
            ctx = BuildContext(demo_context)
            builder = Builder(
                layer_store=store,
                cache=cache,
                context=ctx,
                build_args={"VERSION": "2.0.0"},
            )
            result = builder.build_from_file(demo_dockerfile)
            self.assertTrue(result.success, result.error_message)

            # 检查构建结果
            self.assertEqual(result.final_state.env.get("APP_VERSION"), "2.0.0")
            self.assertEqual(result.final_state.user, "nobody")
            self.assertIn("8080", result.final_state.exposed_ports)
            self.assertIn("/app/data", result.final_state.volumes)

            # 打包三种格式都试试
            pkg = Packager(scratch_dir=os.path.join(ws, "scratch"))
            oci_dir = os.path.join(ws, "oci")
            pr1 = pkg.pack_oci_layout(result, oci_dir, name="demo")
            self.assertTrue(os.path.isdir(oci_dir))

            dtar = os.path.join(ws, "demo.tar")
            pr2 = pkg.pack_docker_tar(result, dtar, name="demo")
            self.assertTrue(os.path.exists(dtar))

            ftar = os.path.join(ws, "demo-flat.tar.gz")
            pr3 = pkg.pack_flat_tar(result, ftar)
            self.assertTrue(os.path.exists(ftar))

            # 验证 materialize 后的文件内容
            out = os.path.join(ws, "fs")
            lfs = LayeredFilesystem(result.layers)
            lfs.materialize(out)
            # hello.txt 应被删除
            self.assertFalse((Path(out) / "app" / "data" / "hello.txt").exists())
            # version.txt 应包含版本号
            vf = Path(out) / "app" / "data" / "version.txt"
            self.assertTrue(vf.exists())
            self.assertIn("2.0.0", vf.read_text())
            # requirements.txt 应被复制
            self.assertTrue((Path(out) / "app" / "requirements.txt").exists())
            # src/app.py 应被复制
            self.assertTrue((Path(out) / "app" / "src" / "app.py").exists())
            # added.txt 来自后一个 RUN
            self.assertTrue((Path(out) / "app" / "added.txt").exists())
        finally:
            shutil.rmtree(ws, ignore_errors=True)

    def test_e2e_variable_expansion(self):
        """端到端: 验证 ARG → ENV → WORKDIR/COPY/VOLUME 全链路变量展开。"""
        ws = tempfile.mkdtemp(prefix="mib-test-e2e-var-")
        try:
            ctx_dir = os.path.join(ws, "ctx")
            os.makedirs(ctx_dir)
            (Path(ctx_dir) / "hello.txt").write_text("hello\n")

            df = """
FROM scratch
ARG VERSION=1.0
ARG PORT=8080
ENV APP_VERSION=${VERSION}
ENV APP_HOME=/opt/app-${APP_VERSION}
WORKDIR ${APP_HOME}
COPY hello.txt ${APP_HOME}/
RUN echo "version=${APP_VERSION}" > ${APP_HOME}/version.txt
VOLUME ${APP_HOME}/data
EXPOSE ${PORT}
USER appuser
LABEL version=${APP_VERSION}
"""
            store = LayerStore(os.path.join(ws, "store"))
            cache = LayerCache(os.path.join(ws, "cache"), store)
            ctx = BuildContext(ctx_dir)
            builder = Builder(
                layer_store=store,
                cache=cache,
                context=ctx,
                build_args={"VERSION": "2.0"},
            )
            result = builder.build_from_dockerfile(df)
            self.assertTrue(result.success, result.error_message)

            # 检查 state 中的变量都是真实值
            state = result.final_state
            self.assertEqual(state.env.get("APP_VERSION"), "2.0")
            self.assertEqual(state.env.get("APP_HOME"), "/opt/app-2.0")
            self.assertEqual(state.workdir, "/opt/app-2.0")
            self.assertEqual(state.user, "appuser")
            self.assertIn("8080", state.exposed_ports)
            self.assertIn("/opt/app-2.0/data", state.volumes)
            self.assertEqual(state.labels.get("version"), "2.0")

            # 验证 materialize 后文件在正确路径下
            out = os.path.join(ws, "fs")
            lfs = LayeredFilesystem(result.layers)
            lfs.materialize(out)
            self.assertTrue((Path(out) / "opt" / "app-2.0" / "hello.txt").exists())
            vf = Path(out) / "opt" / "app-2.0" / "version.txt"
            self.assertTrue(vf.exists())
            self.assertIn("version=2.0", vf.read_text())

            # 打包 OCI 和 flat tar 都验证一下
            pkg = Packager(scratch_dir=os.path.join(ws, "scratch"))
            oci_dir = os.path.join(ws, "oci")
            pkg.pack_oci_layout(result, oci_dir, name="var-test")
            self.assertTrue(os.path.isdir(oci_dir))

            ftar = os.path.join(ws, "flat.tar.gz")
            pkg.pack_flat_tar(result, ftar)
            self.assertTrue(os.path.exists(ftar))

            # 解压 flat tar 验证路径
            flat_extract = os.path.join(ws, "flat-extract")
            os.makedirs(flat_extract)
            with tarfile.open(ftar, "r:gz") as tf:
                tf.extractall(flat_extract)
            self.assertTrue(
                (Path(flat_extract) / "opt" / "app-2.0" / "hello.txt").exists()
            )
        finally:
            shutil.rmtree(ws, ignore_errors=True)

    def test_e2e_add_tar_unpack(self):
        """端到端: 验证 ADD tar 归档直接解到目标目录, 不保留原始 tar。"""
        ws = tempfile.mkdtemp(prefix="mib-test-e2e-add-")
        try:
            ctx_dir = os.path.join(ws, "ctx")
            os.makedirs(ctx_dir)

            # 造一个 tar 包: 含 a.txt, sub/b.txt
            tar_path = os.path.join(ctx_dir, "archive.tar")
            with tarfile.open(tar_path, "w") as tf:
                info = tarfile.TarInfo(name="a.txt")
                info.size = 6
                tf.addfile(info, fileobj=__import__("io").BytesIO(b"hello\n"))
                info2 = tarfile.TarInfo(name="sub/")
                info2.type = tarfile.DIRTYPE
                tf.addfile(info2)
                info3 = tarfile.TarInfo(name="sub/b.txt")
                info3.size = 6
                tf.addfile(info3, fileobj=__import__("io").BytesIO(b"world\n"))

            df = """
FROM scratch
WORKDIR /data
ADD archive.tar /data/
RUN ls /data > /tmp/list.txt
"""
            store = LayerStore(os.path.join(ws, "store"))
            cache = LayerCache(os.path.join(ws, "cache"), store)
            ctx = BuildContext(ctx_dir)
            builder = Builder(layer_store=store, cache=cache, context=ctx)
            result = builder.build_from_dockerfile(df)
            self.assertTrue(result.success, result.error_message)

            # 验证 materialize 结果
            out = os.path.join(ws, "fs")
            lfs = LayeredFilesystem(result.layers)
            lfs.materialize(out)

            # a.txt 和 sub/b.txt 应该存在
            self.assertTrue((Path(out) / "data" / "a.txt").exists())
            self.assertTrue((Path(out) / "data" / "sub" / "b.txt").exists())
            # 原始 tar 包不应该存在
            self.assertFalse((Path(out) / "data" / "archive.tar").exists())
            self.assertFalse((Path(out) / "archive.tar").exists())
            # 文件内容正确
            self.assertEqual(
                (Path(out) / "data" / "a.txt").read_text(), "hello\n"
            )
            self.assertEqual(
                (Path(out) / "data" / "sub" / "b.txt").read_text(), "world\n"
            )

            # 验证 OCI 打包后层中没有原始 tar
            pkg = Packager(scratch_dir=os.path.join(ws, "scratch"))
            oci_dir = os.path.join(ws, "oci")
            pkg.pack_oci_layout(result, oci_dir, name="add-test")
            # 检查所有层的 tar 中都不含 archive.tar
            import json as _json
            idx_path = os.path.join(oci_dir, "index.json")
            with open(idx_path) as f:
                idx = _json.load(f)
            manifest_digest = idx["manifests"][0]["digest"].split(":")[1]
            manifest_path = os.path.join(
                oci_dir, "blobs", "sha256", manifest_digest
            )
            with open(manifest_path) as f:
                manifest = _json.load(f)
            for layer_desc in manifest["layers"]:
                layer_digest = layer_desc["digest"].split(":")[1]
                layer_blob = os.path.join(
                    oci_dir, "blobs", "sha256", layer_digest
                )
                with tarfile.open(layer_blob, "r") as ltf:
                    names = ltf.getnames()
                    self.assertNotIn(
                        "archive.tar",
                        names,
                        f"层 {layer_digest[:12]} 中不应包含原始 tar 包",
                    )

            # 验证 flat tar 结果
            ftar = os.path.join(ws, "flat.tar.gz")
            pkg.pack_flat_tar(result, ftar)
            flat_extract = os.path.join(ws, "flat-extract")
            os.makedirs(flat_extract)
            with tarfile.open(ftar, "r:gz") as tf:
                tf.extractall(flat_extract)
            self.assertTrue(
                (Path(flat_extract) / "data" / "a.txt").exists()
            )
            self.assertTrue(
                (Path(flat_extract) / "data" / "sub" / "b.txt").exists()
            )
            self.assertFalse(
                (Path(flat_extract) / "data" / "archive.tar").exists()
            )
        finally:
            shutil.rmtree(ws, ignore_errors=True)

    def test_e2e_directory_deletion(self):
        """端到端: 验证 RUN rm -rf /dir 产生目录级 whiteout, 下层内容不冒出。"""
        ws = tempfile.mkdtemp(prefix="mib-test-e2e-rm-")
        try:
            ctx_dir = os.path.join(ws, "ctx")
            os.makedirs(ctx_dir)
            (Path(ctx_dir) / "keep.txt").write_text("keep me\n")

            # 第一层建 /app 目录及文件, 第二层删整个 /app 目录
            df = """
FROM scratch
WORKDIR /app
RUN mkdir -p /app/sub && \
    echo "file1" > /app/file1.txt && \
    echo "file2" > /app/sub/file2.txt && \
    echo "file3" > /app/sub/file3.txt
COPY keep.txt /app/keep.txt
RUN rm -rf /app
RUN mkdir -p /app && echo "new" > /app/new.txt
"""
            store = LayerStore(os.path.join(ws, "store"))
            cache = LayerCache(os.path.join(ws, "cache"), store)
            ctx = BuildContext(ctx_dir)
            builder = Builder(layer_store=store, cache=cache, context=ctx)
            result = builder.build_from_dockerfile(df)
            self.assertTrue(result.success, result.error_message)

            # 验证 materialize 后 /app 里只有新文件, 旧文件都消失
            out = os.path.join(ws, "fs")
            lfs = LayeredFilesystem(result.layers)
            lfs.materialize(out)
            app_dir = Path(out) / "app"
            self.assertTrue(app_dir.exists())
            self.assertTrue((app_dir / "new.txt").exists())
            self.assertEqual((app_dir / "new.txt").read_text(), "new\n")
            # 旧文件全部消失
            self.assertFalse((app_dir / "file1.txt").exists())
            self.assertFalse((app_dir / "keep.txt").exists())
            self.assertFalse((app_dir / "sub").exists())

            # 验证 flat tar 打包结果
            pkg = Packager(scratch_dir=os.path.join(ws, "scratch"))
            ftar = os.path.join(ws, "flat.tar.gz")
            pkg.pack_flat_tar(result, ftar)
            flat_extract = os.path.join(ws, "flat-extract")
            os.makedirs(flat_extract)
            with tarfile.open(ftar, "r:gz") as tf:
                tf.extractall(flat_extract)
            self.assertTrue(
                (Path(flat_extract) / "app" / "new.txt").exists()
            )
            self.assertFalse(
                (Path(flat_extract) / "app" / "file1.txt").exists()
            )
            self.assertFalse(
                (Path(flat_extract) / "app" / "sub").exists()
            )
            self.assertFalse(
                (Path(flat_extract) / "app" / "keep.txt").exists()
            )

            # 验证 OCI 打包后层叠加结果
            oci_dir = os.path.join(ws, "oci")
            pkg.pack_oci_layout(result, oci_dir, name="rm-test")
            # 从 OCI 中读取层并重新构建 LayeredFilesystem 验证
            import json as _json
            idx_path = os.path.join(oci_dir, "index.json")
            with open(idx_path) as f:
                idx = _json.load(f)
            manifest_digest = idx["manifests"][0]["digest"].split(":")[1]
            manifest_path = os.path.join(
                oci_dir, "blobs", "sha256", manifest_digest
            )
            with open(manifest_path) as f:
                manifest = _json.load(f)
            # 找删除层对应的 tar, 检查是否有 .wh.app whiteout 标记
            found_wh = False
            for layer_desc in manifest["layers"]:
                layer_digest = layer_desc["digest"].split(":")[1]
                layer_blob = os.path.join(
                    oci_dir, "blobs", "sha256", layer_digest
                )
                with tarfile.open(layer_blob, "r") as ltf:
                    names = ltf.getnames()
                    if ".wh.app" in names or "app/.wh..wh..opq" in names:
                        found_wh = True
                        break
            self.assertTrue(
                found_wh,
                "删除 /app 目录后层中应该有 .wh.app 或 opaque 标记",
            )
        finally:
            shutil.rmtree(ws, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
