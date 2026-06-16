"""命令行入口。

使用示例:
    mib build -f Dockerfile -c ./src -o image.tar
    mib build -f Dockerfile -o ./oci-out --format oci
    mib inspect ./oci-out
    mib cache --clear
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from .builder import Builder, BuildEventSink, PrintEventSink
from .cache import LayerCache
from .context import BuildContext
from .layer import LayerStore
from .packager import Packager


class _CliEventSink(PrintEventSink):
    """带颜色的 CLI 事件输出。"""

    COLORS = {
        "STEP": "\033[1;36m",
        "CACHE": "\033[1;32m",
        "BUILD": "\033[1;33m",
        "FROM": "\033[1;35m",
        "RUN-EXEC": "\033[36m",
        "RUN-OUT": "\033[2m",
        "COPY": "\033[1;34m",
        "ENV": "\033[1;32m",
        "LABEL": "\033[1;32m",
        "WORKDIR": "\033[1;32m",
        "USER": "\033[1;32m",
        "EXPOSE": "\033[1;32m",
        "VOLUME": "\033[1;32m",
        "CMD": "\033[1;32m",
        "ENTRYPOINT": "\033[1;32m",
        "ARG": "\033[2m",
        "ERROR": "\033[1;31m",
    }
    RESET = "\033[0m"

    def __init__(self, use_color: bool = True) -> None:
        self.use_color = use_color and sys.stdout.isatty()

    def on_event(self, event) -> None:
        color = self.COLORS.get(event.kind, "") if self.use_color else ""
        reset = self.RESET if self.use_color else ""
        prefix = f"Step {event.step_index}/?" if event.step_index is not None else "*"
        print(f"{color}[{prefix}] {event.kind:<8}: {event.message}{reset}")


def _ensure_workspace(root: str) -> dict:
    """创建默认工作区路径 (layer store + cache)。"""
    root_p = Path(root)
    root_p.mkdir(parents=True, exist_ok=True)
    return {
        "layers": str((root_p / "layers").resolve()),
        "cache": str((root_p / "cache").resolve()),
        "scratch": str((root_p / "scratch").resolve()),
    }


def _parse_build_args(items: List[str]) -> dict:
    out = {}
    for item in items or []:
        if "=" not in item:
            out[item] = ""
        else:
            k, _, v = item.partition("=")
            out[k] = v
    return out


def cmd_build(args: argparse.Namespace) -> int:
    # 1. 配置路径
    ws = _ensure_workspace(args.workspace)
    store = LayerStore(ws["layers"])
    cache = LayerCache(ws["cache"], store)

    # 2. 构建上下文
    dockerfile = args.file or "Dockerfile"
    ctx_dir = args.context or os.path.dirname(os.path.abspath(dockerfile))
    ctx = BuildContext(ctx_dir)

    # 3. 构建器
    sink = _CliEventSink(use_color=not args.no_color)
    builder = Builder(
        layer_store=store,
        cache=cache,
        context=ctx,
        event_sink=sink,
        use_cache=not args.no_cache,
        build_args=_parse_build_args(args.build_arg),
    )

    df_path = Path(dockerfile)
    if not df_path.is_absolute():
        df_path = Path.cwd() / df_path
    result = builder.build_from_file(str(df_path))

    if not result.success:
        print(f"\n\033[1;31m构建失败:\033[0m {result.error_message}", file=sys.stderr)
        return 1

    # 4. 打包
    output = args.output or "mib-output"
    packager = Packager(scratch_dir=ws["scratch"])
    name = args.name or "mini-image"
    tag = args.tag or "latest"

    if args.format == "oci":
        presult = packager.pack_oci_layout(result, output, name=name, tag=tag)
    elif args.format == "docker":
        if not output.endswith((".tar", ".tar.gz")):
            output += ".tar"
        presult = packager.pack_docker_tar(result, output, name=name, tag=tag)
    elif args.format == "flat":
        if not output.endswith((".tar", ".tar.gz")):
            output += ".tar.gz"
        presult = packager.pack_flat_tar(result, output)
    else:
        print(f"未知格式: {args.format}", file=sys.stderr)
        return 2

    print()
    print(f"\033[1;32m✓ 构建完成\033[0m")
    print(f"  格式:       {presult.format}")
    print(f"  输出:       {presult.output_path}")
    print(f"  层数 (非空): {len(presult.packed_layers)}")
    total_size = sum(p.size for p in presult.packed_layers)
    print(f"  总大小:     {_human_size(total_size)}")
    for i, p in enumerate(presult.packed_layers):
        desc = (p.layer.created_by or "").split("\n")[0][:60]
        print(f"    L{i+1:02d} {p.layer.layer_id[:12]}  {_human_size(p.size):>8}  {desc}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if path.is_dir() and (path / "oci-layout").exists():
        return _inspect_oci(path)
    elif path.is_file() and path.suffix == ".tar":
        return _inspect_docker_tar(path)
    else:
        print(f"无法识别的镜像路径: {path}", file=sys.stderr)
        return 1


def _inspect_oci(path: Path) -> int:
    index = json.loads((path / "index.json").read_text())
    print("OCI Image Layout")
    print(f"  manifests: {len(index['manifests'])}")
    for m in index["manifests"]:
        print(f"    digest: {m['digest']}")
        print(f"    size:   {m['size']}")
        if "annotations" in m:
            for k, v in m["annotations"].items():
                print(f"    {k}: {v}")
        # 读取 manifest blob
        h = m["digest"].split(":", 1)[1]
        manifest_file = path / "blobs" / "sha256" / h
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text())
            ch = manifest["config"]["digest"].split(":", 1)[1]
            config_file = path / "blobs" / "sha256" / ch
            if config_file.exists():
                cfg = json.loads(config_file.read_text())
                print(f"  config.Env: {cfg.get('config', {}).get('Env', [])}")
                print(f"  config.Cmd: {cfg.get('config', {}).get('Cmd')}")
                print(f"  rootfs.diff_ids ({len(cfg['rootfs']['diff_ids'])} layers):")
                for d in cfg["rootfs"]["diff_ids"]:
                    print(f"    - {d}")
    return 0


def _inspect_docker_tar(path: Path) -> int:
    import tarfile
    with tarfile.open(path, "r") as tf:
        names = tf.getnames()
        mf = [n for n in names if os.path.basename(n) == "manifest.json"]
        if not mf:
            print("无 manifest.json, 不是有效的 Docker tar", file=sys.stderr)
            return 1
        with tf.extractfile(mf[0]) as f:
            manifest_list = json.loads(f.read())
        for m in manifest_list:
            print(f"  RepoTags: {m.get('RepoTags', [])}")
            print(f"  Config:   {m['Config']}")
            print(f"  Layers:   {len(m['Layers'])}")
            for l in m["Layers"]:
                print(f"    - {l}")
    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    ws = _ensure_workspace(args.workspace)
    store = LayerStore(ws["layers"])
    cache = LayerCache(ws["cache"], store)
    if args.clear:
        cache.clear()
        print("已清空缓存")
        return 0
    stats = cache.stats()
    layers = store.list_layers()
    total_size = 0
    for l in layers:
        try:
            for root, _, files in os.walk(l.root):
                for fn in files:
                    total_size += os.path.getsize(os.path.join(root, fn))
        except Exception:
            pass
    print(f"缓存条目: {stats['entries']}")
    print(f"层总数:   {len(layers)}")
    print(f"总大小:   {_human_size(total_size)}")
    return 0


def _human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024
        i += 1
    return f"{f:.1f} {units[i]}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mib",
        description="Mini Image Builder - 轻量级容器镜像构建工具",
    )
    p.add_argument("--workspace", default=os.path.expanduser("~/.mini-builder"), help="工作区目录 (默认 ~/.mini-builder)")
    p.add_argument("--no-color", action="store_true", help="关闭彩色输出")
    sub = p.add_subparsers(dest="command", required=True)

    # build
    bp = sub.add_parser("build", help="构建镜像")
    bp.add_argument("-f", "--file", help="Dockerfile 路径")
    bp.add_argument("-c", "--context", help="构建上下文目录 (默认 Dockerfile 所在目录)")
    bp.add_argument("-o", "--output", help="输出路径")
    bp.add_argument("--format", choices=["oci", "docker", "flat"], default="oci", help="输出格式")
    bp.add_argument("-t", "--tag", default="latest", help="镜像标签")
    bp.add_argument("-n", "--name", default="mini-image", help="镜像名称")
    bp.add_argument("--build-arg", action="append", default=[], help="构建参数 K=V")
    bp.add_argument("--no-cache", action="store_true", help="禁用缓存")

    # inspect
    ip = sub.add_parser("inspect", help="检查已打包镜像")
    ip.add_argument("path", help="镜像路径 (OCI 目录或 Docker tar)")

    # cache
    cp = sub.add_parser("cache", help="管理缓存")
    cp.add_argument("--clear", action="store_true", help="清空缓存")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "build":
        return cmd_build(args)
    elif args.command == "inspect":
        return cmd_inspect(args)
    elif args.command == "cache":
        return cmd_cache(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
