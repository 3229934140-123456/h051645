"""构建上下文模块。

构建上下文如何传入:
    1. 用户通过 CLI 的 -f <Dockerfile> 和 -c <context_dir> 指定
    2. BuildContext 接收一个本地目录路径, 该目录被视为「上下文根」
    3. COPY/ADD 指令中的源路径都相对于该根, 且无法逃逸 (类似 chroot)
    4. 为了缓存命中, BuildContext 会对每个文件内容做 sha256 并生成
      「上下文清单」(manifest), 该清单哈希可混入构建步骤的缓存 key。
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class ContextFile:
    """上下文中的一个文件或目录条目。"""

    rel_path: str
    abs_path: str
    is_dir: bool
    size: int
    sha256: str


class BuildContext:
    """封装构建上下文目录。

    典型用法::

        ctx = BuildContext("/path/to/app")
        ctx.resolve("src/main.py")  # -> Path("/path/to/app/src/main.py")
        ctx.file_hash("src/main.py")  # -> sha256 hex

    安全保证: resolve 会拒绝包含 ``..`` 的路径, 防止 COPY ../etc/passwd 越界。
    """

    def __init__(self, root_dir: str | os.PathLike) -> None:
        self.root: Path = Path(root_dir).resolve()
        if not self.root.is_dir():
            raise ValueError(f"构建上下文不是目录: {self.root}")
        self._manifest: Optional[Dict[str, ContextFile]] = None
        self._cached_total_hash: Optional[str] = None

    # ------------------------------------------------------------------
    # 路径解析
    # ------------------------------------------------------------------

    def resolve(self, rel_path: str) -> Path:
        """将相对路径解析为绝对路径, 并做安全校验。

        Raises:
            ValueError: 路径逃逸出上下文根
        """
        p = Path(rel_path)
        if p.is_absolute():
            # 把绝对路径转换为相对于根的表达, 例如 /etc/hosts -> root/etc/hosts
            p = Path(p.as_posix().lstrip("/"))
        resolved = (self.root / p).resolve()
        # 安全检查: 必须在 root 之下
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise ValueError(
                f"路径越界: {rel_path!r} 解析为 {resolved} 超出上下文根 {self.root}"
            )
        return resolved

    def exists(self, rel_path: str) -> bool:
        try:
            return self.resolve(rel_path).exists()
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # 内容哈希
    # ------------------------------------------------------------------

    @staticmethod
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def file_hash(self, rel_path: str) -> str:
        """返回单个文件的 sha256。"""
        p = self.resolve(rel_path)
        if not p.is_file():
            return ""
        return self._sha256_file(p)

    # ------------------------------------------------------------------
    # 清单 (manifest) - 所有文件的哈希快照, 用于缓存
    # ------------------------------------------------------------------

    def _build_manifest(self) -> Dict[str, ContextFile]:
        m: Dict[str, ContextFile] = {}
        for dirpath, dirnames, filenames in os.walk(self.root):
            rel_dir = os.path.relpath(dirpath, self.root)
            if rel_dir == ".":
                rel_dir = ""
            for d in dirnames:
                rel = os.path.join(rel_dir, d).replace("\\", "/") if rel_dir else d
                abs_p = os.path.join(dirpath, d)
                m[rel] = ContextFile(
                    rel_path=rel, abs_path=abs_p, is_dir=True, size=0, sha256=""
                )
            for fn in filenames:
                rel = os.path.join(rel_dir, fn).replace("\\", "/") if rel_dir else fn
                abs_p = os.path.join(dirpath, fn)
                try:
                    size = os.path.getsize(abs_p)
                    sha = self._sha256_file(Path(abs_p))
                except OSError:
                    continue
                m[rel] = ContextFile(
                    rel_path=rel, abs_path=abs_p, is_dir=False, size=size, sha256=sha
                )
        return m

    @property
    def manifest(self) -> Dict[str, ContextFile]:
        if self._manifest is None:
            self._manifest = self._build_manifest()
        return self._manifest

    def content_hash_for(self, rel_paths: Iterable[str]) -> str:
        """给定一组相对路径, 计算合并内容哈希。

        这是 COPY/ADD 指令缓存 key 的关键组成部分:
        相同源文件列表 + 相同内容 → 相同哈希 → 缓存命中。
        """
        h = hashlib.sha256()
        for rp in sorted(rel_paths):
            entry = self.manifest.get(rp)
            if entry is None:
                h.update(f"MISSING:{rp}\n".encode())
            elif entry.is_dir:
                # 目录: 递归其下所有文件
                subkeys = sorted(
                    k for k in self.manifest if k.startswith(rp + "/")
                )
                for sk in subkeys:
                    e = self.manifest[sk]
                    h.update(f"{sk}|{e.size}|{e.sha256}\n".encode())
            else:
                h.update(f"{entry.rel_path}|{entry.size}|{entry.sha256}\n".encode())
        return h.hexdigest()

    def total_hash(self) -> str:
        """整个上下文的单一哈希, 用于没有明确源路径的场景。"""
        if self._cached_total_hash is None:
            h = hashlib.sha256()
            for k in sorted(self.manifest):
                e = self.manifest[k]
                h.update(f"{k}|{e.is_dir}|{e.size}|{e.sha256}\n".encode())
            self._cached_total_hash = h.hexdigest()
        return self._cached_total_hash

    def collect_paths(self, patterns: Iterable[str]) -> List[str]:
        """根据模式 (字面量或简单 glob) 返回匹配的相对路径列表。"""
        import fnmatch

        result: List[str] = []
        all_keys = list(self.manifest.keys())
        for pat in patterns:
            pat = pat.replace("\\", "/").rstrip("/")
            if not pat:
                continue
            if any(c in pat for c in "*?["):
                matched = [k for k in all_keys if fnmatch.fnmatch(k, pat)]
                result.extend(matched)
            else:
                if pat in self.manifest:
                    result.append(pat)
                # 作为目录前缀, 匹配子项 (即使 pat 本身不在 manifest 中, 也尝试匹配其子内容)
                sub = [k for k in all_keys if k.startswith(pat + "/")]
                result.extend(sub)
                # pat 本身作为目录条目 (若其下有子项, 则也视作匹配)
                if sub and pat not in result:
                    # 检查 pat 是否是目录 (有子项则是)
                    if any(k == pat for k in all_keys):
                        result.append(pat)
        # 去重并排序
        return sorted(set(result))


class EmptyContext(BuildContext):
    """无文件的空上下文, 用于只有 RUN/ENV 等指令的场景测试。"""

    def __init__(self) -> None:
        from tempfile import mkdtemp

        super().__init__(mkdtemp(prefix="mib-empty-ctx-"))
