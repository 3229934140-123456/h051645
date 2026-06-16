"""镜像打包模块。

最终镜像如何组织成「层加配置加清单」的结构:

本模块支持两种常见的镜像分发格式:

1. OCI Image Layout (目录形式)::

    <output_dir>/
        oci-layout              # { "imageLayoutVersion": "1.0.0" }
        index.json              # 入口索引, 引用 manifest
        blobs/
            sha256/
                <manifest-hash>                 # image manifest (JSON)
                <config-hash>                   # image config (JSON, 含 history + rootfs.diff_ids)
                <layer1-hash>.tar.gz            # 第一层 FS 变更
                <layer2-hash>.tar.gz            # 第二层 ...
                ...

2. Docker Tar Archive (docker save 形式, 便于 docker load 导入)::

    <output>.tar
        <layer-id>/
            layer.tar           # 未压缩 FS 变更 (Docker 兼容)
            json                # 该层元数据
        manifest.json           # 顶层清单: [{ "Config":..., "RepoTags":..., "Layers":[...] }]
        <config-hash>.json      # 镜像配置 (与 OCI config 结构相同)
        repositories            # 可选, 名称→层链索引

构建完成后, 打包流程:
    a. 对每个非空层调用 layer.pack_to_tar(), 得到 tar.gz 及其 digest/diff_id/size
    b. 根据构建状态生成 image config (含 rootfs.diff_ids 数组和 history 数组)
    c. 生成 image manifest (含 layers 数组, 每个元素包含 digest/size/mediaType)
    d. 生成 index.json (OCI) 或 manifest.json (Docker)
    e. 把所有 blob 写入 blobs/sha256/<hash>
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .builder import BuildResult, BuildState
from .layer import Layer


MEDIATYPE_OCI_MANIFEST_V1 = "application/vnd.oci.image.manifest.v1+json"
MEDIATYPE_OCI_CONFIG_V1 = "application/vnd.oci.image.config.v1+json"
MEDIATYPE_OCI_LAYER_GZ = "application/vnd.oci.image.layer.v1.tar+gzip"
MEDIATYPE_OCI_INDEX_V1 = "application/vnd.oci.image.index.v1+json"

MEDIATYPE_DOCKER_MANIFEST_V2 = "application/vnd.docker.distribution.manifest.v2+json"
MEDIATYPE_DOCKER_CONFIG_V1 = "application/vnd.docker.container.image.v1+json"
MEDIATYPE_DOCKER_LAYER_GZ = "application/vnd.docker.image.rootfs.diff.tar.gzip"


@dataclass
class PackedLayer:
    layer: Layer
    tar_path: str
    digest: str
    diff_id: str
    size: int

    @property
    def hash_hex(self) -> str:
        return self.digest.split(":", 1)[1]


@dataclass
class PackResult:
    format: str
    output_path: str
    manifest: dict
    config: dict
    packed_layers: List[PackedLayer]

    def as_dict(self) -> dict:
        return {
            "format": self.format,
            "output": self.output_path,
            "layers": [
                {
                    "id": pl.layer.layer_id,
                    "digest": pl.digest,
                    "diff_id": pl.diff_id,
                    "size": pl.size,
                }
                for pl in self.packed_layers
            ],
            "manifest": self.manifest,
            "config": self.config,
        }


class Packager:
    """将 BuildResult 打包为 OCI/Docker 格式。"""

    def __init__(self, *, scratch_dir: Optional[str] = None) -> None:
        if scratch_dir:
            Path(scratch_dir).mkdir(parents=True, exist_ok=True)
            self._scratch = str(Path(scratch_dir).resolve())
        else:
            self._scratch = tempfile.mkdtemp(prefix="mib-pack-")

    # ------------------------------------------------------------------
    # 打包各层为 tar.gz
    # ------------------------------------------------------------------

    def _pack_layers(self, layers: List[Layer]) -> List[PackedLayer]:
        packed: List[PackedLayer] = []
        # 顺序: 从底层到顶层
        for idx, layer in enumerate(layers):
            if layer.empty:
                # 空层: 不需要 FS 层包, 但 config 的 history 中要记录
                continue
            tar_name = f"layer-{idx}-{layer.layer_id}.tar.gz"
            tar_path = os.path.join(self._scratch, tar_name)
            if layer.digest and layer.diff_id and layer.size:
                # 之前已打包过, 直接复用 (需确认文件还在)
                # 这里简单处理: 重新打包以保证文件存在
                pass
            digest, diff_id, size = layer.pack_to_tar(tar_path)
            layer.digest = digest
            layer.diff_id = diff_id
            layer.size = size
            packed.append(PackedLayer(layer=layer, tar_path=tar_path, digest=digest, diff_id=diff_id, size=size))
        return packed

    # ------------------------------------------------------------------
    # 生成 image config
    # ------------------------------------------------------------------

    def _make_config(self, result: BuildResult, packed: List[PackedLayer]) -> dict:
        state: BuildState = result.final_state
        diff_ids = [p.diff_id for p in packed]

        # history: 每层 (含空层) 一个条目
        history: List[dict] = []
        for layer in result.layers:
            item = layer.to_manifest_history_item()
            history.append(item)

        config = {
            "created": "1970-01-01T00:00:00Z",
            "architecture": "amd64",
            "os": "linux",
            "config": {
                "Env": [f"{k}={v}" for k, v in self._default_env(state).items()],
                "Cmd": state.cmd if state.cmd is not None else ["/bin/sh"],
                "Entrypoint": state.entrypoint,
                "WorkingDir": state.workdir if state.workdir != "/" else None,
                "User": state.user if state.user != "root" else None,
                "ExposedPorts": {port: {} for port in state.exposed_ports} if state.exposed_ports else None,
                "Volumes": {v: {} for v in state.volumes} if state.volumes else None,
                "Labels": state.labels if state.labels else None,
            },
            "rootfs": {
                "type": "layers",
                "diff_ids": diff_ids,
            },
            "history": history,
        }
        # 清理 None 字段
        config["config"] = {k: v for k, v in config["config"].items() if v is not None}
        if state.maintainer:
            config["author"] = state.maintainer
        return config

    @staticmethod
    def _default_env(state: BuildState) -> Dict[str, str]:
        env = dict(state.env)
        env.setdefault("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
        return env

    # ------------------------------------------------------------------
    # 生成 image manifest
    # ------------------------------------------------------------------

    @staticmethod
    def _make_manifest(config_digest: str, config_size: int, packed: List[PackedLayer], *, oci: bool) -> dict:
        layer_media = MEDIATYPE_OCI_LAYER_GZ if oci else MEDIATYPE_DOCKER_LAYER_GZ
        config_media = MEDIATYPE_OCI_CONFIG_V1 if oci else MEDIATYPE_DOCKER_CONFIG_V1
        manifest_media = MEDIATYPE_OCI_MANIFEST_V1 if oci else MEDIATYPE_DOCKER_MANIFEST_V2
        return {
            "schemaVersion": 2,
            "mediaType": manifest_media,
            "config": {
                "mediaType": config_media,
                "digest": config_digest,
                "size": config_size,
            },
            "layers": [
                {
                    "mediaType": layer_media,
                    "digest": p.digest,
                    "size": p.size,
                }
                for p in packed
            ],
        }

    # ------------------------------------------------------------------
    # 格式 1: OCI Image Layout
    # ------------------------------------------------------------------

    def pack_oci_layout(self, result: BuildResult, output_dir: str, *, name: str = "mini-image", tag: str = "latest") -> PackResult:
        if not result.success:
            raise RuntimeError(f"构建失败, 无法打包: {result.error_message}")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        blobs_dir = out / "blobs" / "sha256"
        blobs_dir.mkdir(parents=True, exist_ok=True)

        # 1. 打包各层
        packed = self._pack_layers(result.layers)
        for p in packed:
            dest = blobs_dir / p.hash_hex
            shutil.copy2(p.tar_path, dest)

        # 2. 生成 config
        cfg = self._make_config(result, packed)
        cfg_bytes = json.dumps(cfg, indent=2).encode("utf-8")
        cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()
        cfg_size = len(cfg_bytes)
        cfg_digest = "sha256:" + cfg_hash
        (blobs_dir / cfg_hash).write_bytes(cfg_bytes)

        # 3. 生成 manifest
        manifest = self._make_manifest(cfg_digest, cfg_size, packed, oci=True)
        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        manifest_digest = "sha256:" + manifest_hash
        (blobs_dir / manifest_hash).write_bytes(manifest_bytes)

        # 4. oci-layout
        (out / "oci-layout").write_text(json.dumps({"imageLayoutVersion": "1.0.0"}))

        # 5. index.json
        index = {
            "schemaVersion": 2,
            "mediaType": MEDIATYPE_OCI_INDEX_V1,
            "manifests": [
                {
                    "mediaType": MEDIATYPE_OCI_MANIFEST_V1,
                    "digest": manifest_digest,
                    "size": len(manifest_bytes),
                    "annotations": {
                        "org.opencontainers.image.ref.name": f"{name}:{tag}",
                    },
                }
            ],
        }
        (out / "index.json").write_text(json.dumps(index, indent=2))

        return PackResult(
            format="oci",
            output_path=str(out.resolve()),
            manifest=manifest,
            config=cfg,
            packed_layers=packed,
        )

    # ------------------------------------------------------------------
    # 格式 2: Docker tar archive (docker load)
    # ------------------------------------------------------------------

    def pack_docker_tar(self, result: BuildResult, output_tar: str, *, name: str = "mini-image", tag: str = "latest") -> PackResult:
        if not result.success:
            raise RuntimeError(f"构建失败, 无法打包: {result.error_message}")

        packed = self._pack_layers(result.layers)

        # 用临时目录先组装内容, 再整体打包
        assemble_dir = tempfile.mkdtemp(prefix="mib-docker-")
        try:
            # 每个非空层一个子目录
            layer_dirs: List[str] = []
            for p in packed:
                layer_dir = os.path.join(assemble_dir, p.layer.layer_id)
                os.makedirs(layer_dir, exist_ok=True)
                # Docker layer.tar 是未压缩的 (我们需要解压)
                layer_tar = os.path.join(layer_dir, "layer.tar")
                _gunzip(p.tar_path, layer_tar)
                # VERSION
                with open(os.path.join(layer_dir, "VERSION"), "w") as f:
                    f.write("1.0\n")
                # json (层元数据)
                parent = p.layer.parent_id
                layer_json = {
                    "id": p.layer.layer_id,
                    "parent": parent,
                    "created": "1970-01-01T00:00:00Z",
                    "container_config": {"Cmd": [p.layer.created_by]},
                    "config": {},
                }
                with open(os.path.join(layer_dir, "json"), "w") as f:
                    json.dump(layer_json, f)
                layer_dirs.append(p.layer.layer_id)

            # 空层也需要在 history 中, 但 Docker tar 不需要空目录
            # Image config
            cfg = self._make_config(result, packed)
            cfg_bytes = json.dumps(cfg, indent=2).encode("utf-8")
            cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()
            cfg_filename = f"{cfg_hash}.json"
            with open(os.path.join(assemble_dir, cfg_filename), "wb") as f:
                f.write(cfg_bytes)

            # manifest.json
            repo_tag = f"{name}:{tag}"
            docker_manifest = [
                {
                    "Config": cfg_filename,
                    "RepoTags": [repo_tag],
                    "Layers": [f"{lid}/layer.tar" for lid in layer_dirs],
                }
            ]
            with open(os.path.join(assemble_dir, "manifest.json"), "w") as f:
                json.dump(docker_manifest, f, indent=2)

            # repositories
            repos: Dict[str, Dict[str, str]] = {
                name: {tag: (layer_dirs[-1] if layer_dirs else "")}
            }
            with open(os.path.join(assemble_dir, "repositories"), "w") as f:
                json.dump(repos, f, indent=2)

            # 打包成 tar
            with tarfile.open(output_tar, "w") as tar:
                tar.add(assemble_dir, arcname="", recursive=True)

            # manifest 用于返回
            manifest = self._make_manifest("sha256:" + cfg_hash, len(cfg_bytes), packed, oci=False)

            return PackResult(
                format="docker-tar",
                output_path=os.path.abspath(output_tar),
                manifest=manifest,
                config=cfg,
                packed_layers=packed,
            )
        finally:
            shutil.rmtree(assemble_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # 格式 3: 扁平 rootfs tar (方便直接检查内容)
    # ------------------------------------------------------------------

    def pack_flat_tar(self, result: BuildResult, output_tar: str) -> PackResult:
        """把叠加后的完整 rootfs 打为单个 tar (调试用, 不是规范镜像)。"""
        if not result.success:
            raise RuntimeError(f"构建失败, 无法打包: {result.error_message}")

        from .layer import LayeredFilesystem
        materialize_dir = tempfile.mkdtemp(prefix="mib-flat-")
        try:
            lfs = LayeredFilesystem(result.layers)
            lfs.materialize(materialize_dir)

            with tarfile.open(output_tar, "w:gz") as tar:
                tar.add(materialize_dir, arcname="", recursive=True)

            packed = self._pack_layers(result.layers)
            cfg = self._make_config(result, packed)
            return PackResult(
                format="flat-tar",
                output_path=os.path.abspath(output_tar),
                manifest={},
                config=cfg,
                packed_layers=packed,
            )
        finally:
            shutil.rmtree(materialize_dir, ignore_errors=True)


def _gunzip(src_path: str, dst_path: str) -> None:
    import gzip
    with open(src_path, "rb") as fin:
        with gzip.open(fin, "rb") as gz:
            with open(dst_path, "wb") as fout:
                while True:
                    chunk = gz.read(1 << 20)
                    if not chunk:
                        break
                    fout.write(chunk)
