"""层管理模块 - 文件系统层与叠加 (union mount 语义)。

本实现用「目录快照」模拟镜像层:
    - 每层都有自己独立的目录 (layer_root), 存放「本层相对上一层的变更」
    - 变更包含:
        1. 新增/修改文件: 常规文件
        2. 删除文件:    用 whiteout 文件 (前缀 ``.wh.``) 标记,
                        例如删除 /etc/foo.conf  →  创建 /etc/.wh.foo.conf
                        删除整个目录 /var/log →  创建 /var/.wh.log
                        「不透明目录」标记:     某目录下创建 .wh..wh..opq
                        表示该目录的下层内容全部被覆盖 (不再继承)
    - 读取时: 将多个层按顺序「叠加」, 下层为基础, 上层遇到 .wh.* 跳过对应
              下层条目, 遇到 .wh..wh..opq 则跳过该目录下所有下层条目。

这与 Docker / OCI 镜像层使用的 AUFS / OverlayFS 语义一致, 但我们用
纯文件系统目录 + tar 打包来实现, 无需内核支持。
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
from typing import Dict, Iterable, List, Optional, Set, Tuple


WHITEOUT_PREFIX = ".wh."
OPAQUE_MARKER = ".wh..wh..opq"


@dataclass
class Layer:
    """单个文件系统层。

    Attributes:
        layer_id:   层的唯一 ID (通常是内容哈希的前 16 位)
        parent_id:  父层 ID, 最底层 (基础层) 为 None
        root:       层内容的本地目录路径 (存放变更文件与 whiteout)
        digest:     内容 sha256 摘要 (打包后的 tar 的哈希, 用于 OCI manifest)
        diff_id:    解压后的 tar 内容哈希 (OCI config 中使用)
        size:       压缩后大小 (字节)
        created_by: 产生本层的构建指令描述 (写入 image config)
        command:    命令参数 (写入 image config)
        empty:      标记空层 (仅修改元数据, 无文件变更)
    """

    layer_id: str
    parent_id: Optional[str]
    root: str
    digest: str = ""
    diff_id: str = ""
    size: int = 0
    created_by: str = ""
    command: str = ""
    empty: bool = False
    _opaque_dirs_cache: Optional[Set[str]] = None
    _whiteout_cache: Optional[Dict[str, bool]] = None

    # ------------------------------------------------------------------
    # whiteout 辅助静态方法
    # ------------------------------------------------------------------

    @staticmethod
    def is_whiteout(rel_path: str) -> bool:
        name = os.path.basename(rel_path)
        return name.startswith(WHITEOUT_PREFIX) and name != OPAQUE_MARKER

    @staticmethod
    def is_opaque_marker(rel_path: str) -> bool:
        return os.path.basename(rel_path) == OPAQUE_MARKER

    @staticmethod
    def whiteout_target(whiteout_rel_path: str) -> str:
        """.wh.foo  →  foo (返回相对路径, 含父目录)。"""
        dirname = os.path.dirname(whiteout_rel_path)
        name = os.path.basename(whiteout_rel_path)
        target_name = name[len(WHITEOUT_PREFIX):]
        return os.path.join(dirname, target_name).replace("\\", "/") if dirname else target_name

    # ------------------------------------------------------------------
    # 写入 API (构建时使用)
    # ------------------------------------------------------------------

    def add_file(self, dest_rel: str, src_abs: str) -> None:
        """将外部文件复制进本层。"""
        dest_rel = dest_rel.lstrip("/").replace("\\", "/")
        dest_abs = os.path.join(self.root, dest_rel)
        os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
        # 如果之前有 whiteout 标记, 先删除 (新的文件覆盖了删除意图)
        self._remove_whiteout(dest_rel)
        shutil.copy2(src_abs, dest_abs)

    def add_directory_tree(self, dest_rel: str, src_dir: str) -> None:
        """把整个目录树复制进本层, 保留相对结构。"""
        dest_rel = dest_rel.lstrip("/").replace("\\", "/")
        for dirpath, _, filenames in os.walk(src_dir):
            rel_dir = os.path.relpath(dirpath, src_dir)
            if rel_dir == ".":
                rel_dir = ""
            target_dir = os.path.join(self.root, dest_rel, rel_dir) if rel_dir else os.path.join(self.root, dest_rel)
            os.makedirs(target_dir, exist_ok=True)
            for fn in filenames:
                src_file = os.path.join(dirpath, fn)
                rel_file = os.path.join(dest_rel, rel_dir, fn).replace("\\", "/") if rel_dir else os.path.join(dest_rel, fn).replace("\\", "/")
                self._remove_whiteout(rel_file)
                shutil.copy2(src_file, os.path.join(target_dir, fn))

    def add_string_content(self, dest_rel: str, content: str, encoding: str = "utf-8") -> None:
        dest_rel = dest_rel.lstrip("/").replace("\\", "/")
        dest_abs = os.path.join(self.root, dest_rel)
        os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
        self._remove_whiteout(dest_rel)
        with open(dest_abs, "w", encoding=encoding) as f:
            f.write(content)

    def remove_path(self, target_rel: str) -> None:
        """用 whiteout 标记删除路径 (不实际删除父层内容, 仅上层标记)。"""
        target_rel = target_rel.lstrip("/").replace("\\", "/")
        # 先取消本层对该路径的任何新增
        self._remove_self_path(target_rel)
        # 然后创建 whiteout 标记
        dirname = os.path.dirname(target_rel)
        basename = os.path.basename(target_rel)
        wh_dir = os.path.join(self.root, dirname) if dirname else self.root
        os.makedirs(wh_dir, exist_ok=True)
        wh_path = os.path.join(wh_dir, WHITEOUT_PREFIX + basename)
        with open(wh_path, "w") as f:
            f.write("")

    def mark_opaque(self, dir_rel: str) -> None:
        """将目录标记为「不透明」, 表示下层该目录下的所有内容被覆盖。"""
        dir_rel = dir_rel.lstrip("/").replace("\\", "/")
        target = os.path.join(self.root, dir_rel, OPAQUE_MARKER)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w") as f:
            f.write("")

    def _remove_whiteout(self, rel_path: str) -> None:
        dirname = os.path.dirname(rel_path)
        basename = os.path.basename(rel_path)
        wh_path = os.path.join(self.root, dirname, WHITEOUT_PREFIX + basename) if dirname else os.path.join(self.root, WHITEOUT_PREFIX + basename)
        if os.path.exists(wh_path):
            os.remove(wh_path)

    def _remove_self_path(self, rel_path: str) -> None:
        abs_p = os.path.join(self.root, rel_path)
        if os.path.isdir(abs_p) and not os.path.islink(abs_p):
            shutil.rmtree(abs_p, ignore_errors=True)
        elif os.path.exists(abs_p):
            os.remove(abs_p)

    # ------------------------------------------------------------------
    # 枚举本层所有变更条目 (含 whiteout)
    # ------------------------------------------------------------------

    def walk_changes(self) -> Iterable[Tuple[str, bool, bool]]:
        """遍历本层的「变更」, 产生 (相对路径, is_dir, is_whiteout_or_opaque)。"""
        for dirpath, dirnames, filenames in os.walk(self.root):
            rel_dir = os.path.relpath(dirpath, self.root)
            if rel_dir == ".":
                rel_dir = ""
            # 目录 (不含 whiteout 标记, 因为 whiteout 是文件)
            for d in dirnames:
                rel = os.path.join(rel_dir, d).replace("\\", "/") if rel_dir else d
                yield (rel, True, False)
            for fn in filenames:
                rel = os.path.join(rel_dir, fn).replace("\\", "/") if rel_dir else fn
                is_special = self.is_whiteout(rel) or self.is_opaque_marker(rel)
                yield (rel, False, is_special)

    # ------------------------------------------------------------------
    # 内容哈希 (用于缓存与层比对)
    # ------------------------------------------------------------------

    def compute_content_hash(self) -> str:
        """计算本层所有变更 (递归目录 + 文件内容 + whiteout) 的哈希。

        注意: 这是「层差异」的哈希, 不包含父层, 用于判断两层变更是否相同。
        相同指令 + 相同上下文 → 应当产生相同内容哈希 → 缓存命中。
        """
        h = hashlib.sha256()
        entries: List[Tuple[str, str, str]] = []
        for rel, is_dir, is_special in self.walk_changes():
            if is_dir:
                entries.append(("D", rel, ""))
            elif is_special:
                entries.append(("W", rel, ""))
            else:
                abs_p = os.path.join(self.root, rel)
                with open(abs_p, "rb") as f:
                    file_h = hashlib.sha256()
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        file_h.update(chunk)
                entries.append(("F", rel, file_h.hexdigest()))
        entries.sort()
        for kind, rel, digest in entries:
            h.update(f"{kind}|{rel}|{digest}\n".encode())
        return h.hexdigest()

    # ------------------------------------------------------------------
    # 打包成 tar (OCI 要求的格式), 同时计算 digest / diff_id / size
    # ------------------------------------------------------------------

    def pack_to_tar(self, tar_path: str) -> Tuple[str, str, int]:
        """将本层打包为 tar.gz, 同时计算 OCI 所需的三个字段。

        Returns:
            (digest, diff_id, size)
            - digest:   sha256(tar.gz 内容)  → manifest.json 中使用
            - diff_id:  sha256(未压缩 tar)   → image config 中使用
            - size:     tar.gz 字节数
        """
        diff_h = hashlib.sha256()  # 未压缩 tar
        # 先写出未压缩 tar 到临时文件, 再压缩, 以同时得到两个哈希
        fd, uncompressed = tempfile.mkstemp(suffix=".tar")
        os.close(fd)
        try:
            with tarfile.open(uncompressed, "w") as tf:
                self._add_layer_root_to_tar(tf)
            with open(uncompressed, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    diff_h.update(chunk)
            diff_id = "sha256:" + diff_h.hexdigest()

            # 压缩
            compressed_h = hashlib.sha256()
            with open(uncompressed, "rb") as fin, open(tar_path, "wb") as fout:
                import gzip

                with gzip.GzipFile(fileobj=fout, mode="wb", mtime=0) as gz:
                    while True:
                        chunk = fin.read(1 << 20)
                        if not chunk:
                            break
                        gz.write(chunk)
            # 读完压缩结果算哈希
            with open(tar_path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    compressed_h.update(chunk)
            digest = "sha256:" + compressed_h.hexdigest()
            size = os.path.getsize(tar_path)
            return digest, diff_id, size
        finally:
            os.remove(uncompressed)

    def _add_layer_root_to_tar(self, tar: tarfile.TarFile) -> None:
        for dirpath, dirnames, filenames in os.walk(self.root):
            rel_dir = os.path.relpath(dirpath, self.root)
            if rel_dir == ".":
                rel_dir = "."
            else:
                rel_dir = rel_dir.replace("\\", "/")
            # 目录条目
            if rel_dir != ".":
                ti = tarfile.TarInfo(name=rel_dir)
                ti.type = tarfile.DIRTYPE
                ti.mode = 0o755
                ti.mtime = 0
                tar.addfile(ti)
            for fn in filenames:
                rel = rel_dir + "/" + fn if rel_dir != "." else fn
                abs_p = os.path.join(dirpath, fn)
                tar.add(abs_p, arcname=rel, recursive=False)
            # 显式排序确保可重复
            dirnames.sort()

    # ------------------------------------------------------------------
    # 元数据
    # ------------------------------------------------------------------

    def to_manifest_history_item(self) -> dict:
        """生成 OCI image config 中 history 数组的一项。"""
        item = {
            "created": "1970-01-01T00:00:00Z",
            "created_by": self.created_by or "",
            "comment": "",
            "empty_layer": self.empty,
        }
        return item


# ---------------------------------------------------------------------------
# 层叠加: 将多层合并为一个「完整文件系统」视图 (用于最终检查与 RUN 模拟)
# ---------------------------------------------------------------------------


class LayeredFilesystem:
    """按顺序叠加多层, 提供「读视图」。

    层叠加 (whiteout 删除标记) 如何表示:
        给定层 [L0, L1, L2] (L0 最底层), 对任何路径 P:
        1. 从顶层 (L2) 向下查找
        2. 若某层 Li 中存在 P 对应的 .wh.P (whiteout), 则 P 视为不存在, 停止查找
        3. 若进入目录 D 时 Li 中存在 D/.wh..wh..opq (opaque), 则该
           目录下所有内容只看 Li 及其以上各层, 忽略下层
        4. 否则最先找到的 P 就是最终结果
    """

    def __init__(self, layers: List[Layer]) -> None:
        # 存储顺序: layers[0] 是最底层 (FROM), layers[-1] 是最顶层
        self.layers: List[Layer] = list(layers)

    def resolve_path(self, abs_path: str) -> Tuple[Optional[str], Optional[int]]:
        """查找叠加后路径实际对应的 (文件所在层的绝对路径, 层索引)。

        若路径被 whiteout 删除或不存在, 返回 (None, None)。
        注意: 若某祖先目录被 whiteout (删除目录), 则该路径也视为不存在。
        """
        p = abs_path.lstrip("/").replace("\\", "/")
        if p == "" or p == ".":
            # 根目录始终存在 (虚拟)
            return None, -1
        # 拆分为组件
        parts = [c for c in p.split("/") if c]
        # 从顶层向下找
        layer_idx = len(self.layers) - 1
        opaque_until_layer = -1  # 遇到 opaque 后只查 >= 此值的层
        while layer_idx > opaque_until_layer:
            layer = self.layers[layer_idx]
            # 检查: 路径本身或任何祖先目录是否被本层 whiteout
            wh_found = False
            # 先检查路径本身的 whiteout
            wh_file = (
                os.path.join(layer.root, os.path.dirname(p), WHITEOUT_PREFIX + os.path.basename(p))
                if os.path.dirname(p)
                else os.path.join(layer.root, WHITEOUT_PREFIX + os.path.basename(p))
            )
            if os.path.exists(wh_file):
                wh_found = True
            # 再逐级检查祖先目录的 whiteout
            if not wh_found:
                for i in range(1, len(parts)):
                    anc = "/".join(parts[:i])
                    anc_parent = os.path.dirname(anc).replace("\\", "/")
                    anc_name = os.path.basename(anc)
                    anc_wh = (
                        os.path.join(layer.root, anc_parent, WHITEOUT_PREFIX + anc_name)
                        if anc_parent
                        else os.path.join(layer.root, WHITEOUT_PREFIX + anc_name)
                    )
                    if os.path.exists(anc_wh):
                        wh_found = True
                        break
            if wh_found:
                return None, None

            # 检查本层是否有该路径
            candidate = os.path.join(layer.root, p)
            if os.path.exists(candidate):
                return candidate, layer_idx
            # 检查中间目录是否被 opaque 截断
            # 从完整路径到父目录逐级检查是否有 opq 标记
            has_opaque = False
            for i in range(len(parts), 0, -1):
                sub = "/".join(parts[:i])
                opq = os.path.join(layer.root, sub, OPAQUE_MARKER)
                if os.path.exists(opq):
                    opaque_until_layer = layer_idx
                    has_opaque = True
                    break
            if not has_opaque:
                layer_idx -= 1
        return None, None

    def materialize(self, target_dir: str) -> None:
        """将整个叠加后的文件系统具体化为 target_dir 下的实际文件。

        用于: 调试、最终导出 (flat tar)、RUN 指令的工作目录准备。
        """
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        seen: Set[str] = set()
        whiteouts: Set[str] = set()
        opaque_dirs: Set[str] = set()
        # 从顶层向下遍历, 顶层优先
        for layer_idx in reversed(range(len(self.layers))):
            layer = self.layers[layer_idx]
            # 收集本层 whiteout / opaque
            layer_wh: Set[str] = set()
            layer_opq: Set[str] = set()
            for rel, is_dir, is_special in layer.walk_changes():
                if Layer.is_opaque_marker(rel):
                    d = os.path.dirname(rel).replace("\\", "/")
                    layer_opq.add(d)
                    opaque_dirs.add(d)
                elif Layer.is_whiteout(rel):
                    tgt = Layer.whiteout_target(rel)
                    layer_wh.add(tgt)
                    whiteouts.add(tgt)

            def _is_whitelisted(rel_path: str) -> bool:
                """检查路径本身或任何祖先是否被 whiteout。"""
                if rel_path in whiteouts:
                    return True
                ancestors = _ancestors(rel_path)
                for anc in ancestors:
                    if anc in whiteouts:
                        return True
                return False

            # 现在添加文件, 跳过已处理、whiteout、opaque 影响的下层内容
            for rel, is_dir, is_special in layer.walk_changes():
                if is_special:
                    continue
                if _is_whitelisted(rel):
                    continue
                if rel in seen:
                    continue
                # 检查是否祖先被 opaque 且本层之下 (上层 opaque 意味着下层该目录被忽略)
                skip_by_opaque = False
                ancestors = _ancestors(rel)
                for anc in ancestors:
                    if anc in opaque_dirs:
                        # 确认这个 opaque 是否由上层 (更高 layer_idx) 设置
                        for li in range(layer_idx + 1, len(self.layers)):
                            opq_file = os.path.join(self.layers[li].root, anc, OPAQUE_MARKER)
                            if os.path.exists(opq_file):
                                skip_by_opaque = True
                                break
                        if skip_by_opaque:
                            break
                if skip_by_opaque:
                    continue
                # 执行复制
                src = os.path.join(layer.root, rel)
                dst = os.path.join(target, rel)
                if is_dir:
                    os.makedirs(dst, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                seen.add(rel)


def _ancestors(rel_path: str) -> List[str]:
    parts = [c for c in rel_path.replace("\\", "/").split("/") if c]
    out: List[str] = []
    for i in range(1, len(parts)):
        out.append("/".join(parts[:i]))
    return out


# ---------------------------------------------------------------------------
# 层存储: 管理所有层的创建、持久化
# ---------------------------------------------------------------------------


class LayerStore:
    """层的本地存储。

    目录结构::

        <store_root>/
            layers/
                <layer_id>/
                    layer.json        # 元数据
                    root/             # 变更文件
                    content_hash      # 本层内容哈希 (字符串缓存)
    """

    def __init__(self, store_root: str | os.PathLike) -> None:
        self.store_root = Path(store_root)
        self.layers_dir = self.store_root / "layers"
        self.store_root.mkdir(parents=True, exist_ok=True)
        self.layers_dir.mkdir(exist_ok=True)

    def create_layer(self, parent: Optional[Layer], content_hash: str = "") -> Layer:
        """创建一个新层并分配唯一 ID, 返回 Layer 对象 (已落盘目录)。"""
        layer_id = _make_layer_id(parent.layer_id if parent else "", content_hash or str(id(self)))
        layer_root = self.layers_dir / layer_id
        layer_root.mkdir(parents=True, exist_ok=False)
        (layer_root / "root").mkdir(exist_ok=False)
        layer = Layer(
            layer_id=layer_id,
            parent_id=parent.layer_id if parent else None,
            root=str((layer_root / "root").resolve()),
        )
        self._save_metadata(layer)
        return layer

    def store_layer(self, layer: Layer) -> None:
        """将构建好的层 (修改过 root) 的元数据持久化。"""
        layer_meta_path = self.layers_dir / layer.layer_id
        if not layer_meta_path.exists():
            layer_meta_path.mkdir(parents=True, exist_ok=True)
            (layer_meta_path / "root").mkdir(exist_ok=True)
        self._save_metadata(layer)

    def get_layer(self, layer_id: str) -> Optional[Layer]:
        meta_file = self.layers_dir / layer_id / "layer.json"
        if not meta_file.exists():
            return None
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        return Layer(
            layer_id=data["layer_id"],
            parent_id=data.get("parent_id"),
            root=data["root"],
            digest=data.get("digest", ""),
            diff_id=data.get("diff_id", ""),
            size=data.get("size", 0),
            created_by=data.get("created_by", ""),
            command=data.get("command", ""),
            empty=data.get("empty", False),
        )

    def list_layers(self) -> List[Layer]:
        out: List[Layer] = []
        if not self.layers_dir.exists():
            return out
        for entry in self.layers_dir.iterdir():
            if entry.is_dir() and (entry / "layer.json").exists():
                out.append(self.get_layer(entry.name))
        return [l for l in out if l is not None]

    def _save_metadata(self, layer: Layer) -> None:
        data = {
            "layer_id": layer.layer_id,
            "parent_id": layer.parent_id,
            "root": layer.root,
            "digest": layer.digest,
            "diff_id": layer.diff_id,
            "size": layer.size,
            "created_by": layer.created_by,
            "command": layer.command,
            "empty": layer.empty,
        }
        meta_file = self.layers_dir / layer.layer_id / "layer.json"
        meta_file.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _make_layer_id(parent_salt: str, content_salt: str) -> str:
    h = hashlib.sha256()
    h.update(parent_salt.encode())
    h.update(b"\x00")
    h.update(content_salt.encode())
    h.update(str(os.getpid()).encode())
    import time
    h.update(str(time.time_ns()).encode())
    return h.hexdigest()[:16]
