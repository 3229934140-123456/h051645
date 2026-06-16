"""层缓存模块。

层的内容如何用哈希标识以便相同指令与上下文时命中缓存跳过重建:
    每条构建步骤 Step 的「缓存 key」由以下部分构成:
        chain_hash  = 父层最终的 chain_hash (最底层为 "")
        step_sig    = 规范化后的指令文本 + 类型
        context_sig = 若该步骤涉及文件 (COPY/ADD), 则附加源文件内容哈希
        cache_key   = sha256(chain_hash + "|" + step_sig + "|" + context_sig)

    如果缓存层中存在该 cache_key → 缓存命中, 直接复用该层, 跳过构建。
    如果缓存未命中 → 执行构建, 产生新层, 并将 (cache_key → layer_id) 存入缓存。

缓存如何在某层失效后让其后所有层都失效:
    chain_hash 的链式结构天然保证了这一点:
        假设构建链为: FROM → S1 → S2 → S3
        各层 cache_key 分别为:
            CK1 = sha256(""   | sig(S1))
            CK2 = sha256(CK1  | sig(S2))
            CK3 = sha256(CK2  | sig(S3))
        如果 S1 变更, 则 CK1 改变 → CK2 的输入改变 → CK2 也改变 → CK3 同理。
    因此只要某一步骤的「输入」变化 (指令或上下文), 其后所有步骤的 cache_key
    都会自然变化, 表现为「整条链从该步骤起全部失效」。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .context import BuildContext
from .layer import Layer, LayerStore
from .parser import BuildStep, InstructionType


@dataclass
class CacheEntry:
    """单条缓存记录。"""

    cache_key: str
    layer_id: str
    chain_hash: str          # 本层构建完成后的 chain_hash, 作为下一步骤的输入
    prev_chain_hash: str     # 本层构建前的 chain_hash (即父层输出)


class LayerCache:
    """基于 (chain_hash + 步骤签名) 的层缓存。

    目录结构::

        <cache_root>/
            index.json         # {cache_key: CacheEntry}
            aliases.json       # {layer_id: cache_key} 反向索引
    """

    def __init__(self, cache_root: str | os.PathLike, layer_store: LayerStore) -> None:
        self.cache_root = Path(cache_root)
        self.layer_store = layer_store
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.cache_root / "index.json"
        self._aliases_path = self.cache_root / "aliases.json"
        self._index: Dict[str, dict] = {}
        self._aliases: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self._index_path.exists():
            self._index = json.loads(self._index_path.read_text(encoding="utf-8"))
        if self._aliases_path.exists():
            self._aliases = json.loads(self._aliases_path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self._index_path.write_text(json.dumps(self._index, indent=2), encoding="utf-8")
        self._aliases_path.write_text(json.dumps(self._aliases, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # 核心: 计算 step 的缓存 key
    # ------------------------------------------------------------------

    @staticmethod
    def compute_step_signature(
        step: BuildStep,
        context: Optional[BuildContext],
    ) -> str:
        """计算步骤签名 (不包含父层 chain_hash)。

        对 COPY/ADD 类指令, 会额外混入源文件内容哈希, 因此内容变更会
        导致签名变更 → 缓存失效。
        """
        base = step.cache_key()
        h = hashlib.sha256()
        h.update(base.encode())
        # 对不同类型指令附加额外上下文
        if step.type in (InstructionType.COPY, InstructionType.ADD):
            if context is not None and step.args:
                # args[:-1] 是所有源, args[-1] 是目标
                srcs = step.args[:-1]
                matched_paths = context.collect_paths(srcs)
                content_hash = context.content_hash_for(matched_paths)
                h.update(b"|FILES:")
                h.update(content_hash.encode())
            # --from 标志也影响 COPY 的结果
            if "from" in step.kwargs:
                h.update(f"|FROM:{step.kwargs['from']}".encode())
        elif step.type == InstructionType.RUN:
            # RUN: 主要由其命令字符串决定
            # 注意: 真实构建器还会混入 /etc/resolv.conf 等 host 相关内容
            # 这里只使用规范化的命令内容
            pass
        elif step.type in (InstructionType.ENV, InstructionType.LABEL):
            # 按 key 排序后再混入, 保证等价 dict 产生相同签名
            for k in sorted(step.kwargs):
                h.update(f"|KV:{k}={step.kwargs[k]}".encode())
        return h.hexdigest()

    @staticmethod
    def make_cache_key(prev_chain_hash: str, step_sig: str) -> Tuple[str, str]:
        """根据父层 chain_hash + 步骤签名, 生成缓存 key 与本层 chain_hash。

        Returns:
            (cache_key, next_chain_hash)
            - cache_key:      用于查找缓存
            - next_chain_hash: 作为下一步骤输入的 chain_hash (也即本层输出的链哈希)
        """
        material = f"{prev_chain_hash}|{step_sig}"
        cache_key = hashlib.sha256(("CK:" + material).encode()).hexdigest()
        next_chain = hashlib.sha256(("CH:" + material).encode()).hexdigest()
        return cache_key, next_chain

    # ------------------------------------------------------------------
    # 查询 / 写入 API
    # ------------------------------------------------------------------

    def lookup(self, cache_key: str) -> Optional[CacheEntry]:
        raw = self._index.get(cache_key)
        if raw is None:
            return None
        return CacheEntry(
            cache_key=raw["cache_key"],
            layer_id=raw["layer_id"],
            chain_hash=raw["chain_hash"],
            prev_chain_hash=raw["prev_chain_hash"],
        )

    def get_layer_if_cached(
        self,
        prev_chain_hash: str,
        step: BuildStep,
        context: Optional[BuildContext],
    ) -> Optional[Tuple[Layer, str]]:
        """尝试在缓存中查找该步骤可复用的层。

        Returns:
            (layer, next_chain_hash) 或 None
        """
        step_sig = self.compute_step_signature(step, context)
        cache_key, next_chain = self.make_cache_key(prev_chain_hash, step_sig)
        entry = self.lookup(cache_key)
        if entry is None:
            return None
        layer = self.layer_store.get_layer(entry.layer_id)
        if layer is None:
            # 缓存引用的层已不存在, 清理这条记录
            del self._index[cache_key]
            if self._aliases.get(entry.layer_id) == cache_key:
                del self._aliases[entry.layer_id]
            self._save()
            return None
        return layer, next_chain

    def record(
        self,
        prev_chain_hash: str,
        step: BuildStep,
        context: Optional[BuildContext],
        layer: Layer,
    ) -> str:
        """将 (步骤 → 层) 关联写入缓存。

        Returns:
            next_chain_hash: 该步骤对应的后续 chain_hash
        """
        step_sig = self.compute_step_signature(step, context)
        cache_key, next_chain = self.make_cache_key(prev_chain_hash, step_sig)
        entry = {
            "cache_key": cache_key,
            "layer_id": layer.layer_id,
            "chain_hash": next_chain,
            "prev_chain_hash": prev_chain_hash,
        }
        self._index[cache_key] = entry
        self._aliases[layer.layer_id] = cache_key
        self._save()
        return next_chain

    def invalidate(self, cache_key: str) -> bool:
        if cache_key in self._index:
            entry = self._index.pop(cache_key)
            self._aliases.pop(entry["layer_id"], None)
            self._save()
            return True
        return False

    def clear(self) -> None:
        self._index.clear()
        self._aliases.clear()
        self._save()

    # ------------------------------------------------------------------
    # 统计 / 调试
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {"entries": len(self._index)}
