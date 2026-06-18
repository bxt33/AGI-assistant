"""长期记忆：支持语义向量召回（embedding 优先）或 TF 词袋降级。
核心特性：
  - StoreClassified: 写入前自动 dedup (cosine >= DedupThreshold)
  - RecallByFilter: 受 SlotFilter 约束的召回
  - Consolidate: 合并、衰减、过期淘汰
"""

import math
import threading
import time
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class Item:
    id: int = -1
    content: str = ""
    importance: float = 0.5
    embedding: List[float] = field(default_factory=list)
    created_at: float = 0.0
    last_accessed: float = 0.0
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    slot_hint: str = ""
    score: float = 0.0  # 综合召回分
    tf_vector: Dict[str, float] = field(default_factory=dict)


@dataclass
class RecallFilter:
    categories: List[str] = field(default_factory=list)
    require_tags: List[str] = field(default_factory=list)
    min_score: float = 0.4
    top_k: int = 3
    max_age_hours: int = 0


class LongTerm:
    """长期记忆：支持语义向量 / TF 词袋双层召回"""

    def __init__(self, cfg=None):
        self._mu = threading.RLock()
        self._items: List[Item] = []
        self._next_id = 1
        self._vocab: Dict[str, int] = {}
        self._store_count = 0
        self._cfg = cfg

    def store_item(self, item: Item):
        """直接存储一条记忆（从 PG 恢复时使用），不设置 ID 以让 StoreClassified 生成"""
        with self._mu:
            if item.id < 0:
                item.id = self._next_id
                self._next_id += 1
            else:
                if item.id >= self._next_id:
                    self._next_id = item.id + 1
            if item.created_at == 0:
                item.created_at = time.time()
            if item.last_accessed == 0:
                item.last_accessed = time.time()
            self._items.append(item)

    def store_classified(
        self, content: str, importance: float = 0.5,
        embedding: Optional[List[float]] = None,
        category: str = "general", tags: List[str] = None,
        slot_hint: str = ""
    ) -> Optional[Item]:
        """写入前自动 dedup"""
        with self._mu:
            tags = tags or []
            emb = embedding or []

            # Dedup: cosine >= DedupThreshold 则更新已有条目
            dedup_threshold = 0.95
            if self._cfg:
                dedup_threshold = self._cfg.MemoryConsolidationDedup

            if emb:
                for existing in self._items:
                    if existing.embedding:
                        sim = self._cosine(emb, existing.embedding)
                        if sim >= dedup_threshold:
                            existing.importance = max(existing.importance, importance)
                            existing.last_accessed = time.time()
                            existing.content = content  # 更新为最新内容
                            return existing

            new_item = Item(
                id=self._next_id,
                content=content,
                importance=importance,
                embedding=emb,
                created_at=time.time(),
                last_accessed=time.time(),
                category=category,
                tags=tags,
                slot_hint=slot_hint,
            )
            self._next_id += 1
            self._items.append(new_item)
            self._store_count += 1
            return new_item

    def recall(self, query: str, query_embedding: Optional[List[float]] = None,
               top_k: int = 3) -> List[Item]:
        """语义召回（embedding 优先，TF 词袋降级）"""
        return self.recall_by_filter(query, query_embedding, RecallFilter(top_k=top_k))

    def recall_by_filter(self, query: str, query_embedding: Optional[List[float]],
                         flt: RecallFilter) -> List[Item]:
        with self._mu:
            candidates: List[Item] = []

            # 分类过滤
            if flt.categories:
                cat_set = set(flt.categories)
                items = [it for it in self._items if it.category in cat_set]
            else:
                items = list(self._items)

            # 标签过滤
            if flt.require_tags:
                tag_set = set(flt.require_tags)
                items = [it for it in items if tag_set.issubset(set(it.tags))]

            # 时间过滤
            if flt.max_age_hours > 0:
                cutoff = time.time() - flt.max_age_hours * 3600
                items = [it for it in items if it.created_at >= cutoff]

            if not items:
                return []

            if query_embedding:
                # Embedding 模式：余弦相似度 + 重要性加权
                for it in items:
                    if it.embedding:
                        sim = self._cosine(query_embedding, it.embedding)
                        score = sim * 0.7 + it.importance * 0.3
                        if score >= flt.min_score:
                            it.score = score
                            candidates.append(it)
            else:
                # TF 降级模式
                query_tf = self._text_to_tf(query)
                for it in items:
                    if it.tf_vector:
                        sim = self._cosine_tf(query_tf, it.tf_vector)
                    else:
                        sim = 0.0
                    score = sim * 0.7 + it.importance * 0.3
                    if score >= flt.min_score:
                        it.score = score
                        candidates.append(it)

            # 按分数排序截断
            candidates.sort(key=lambda x: x.score, reverse=True)
            if flt.top_k > 0:
                candidates = candidates[:flt.top_k]

            # 更新访问时间
            for it in candidates:
                it.last_accessed = time.time()

            return candidates

    def filter_by_category(self, categories: List[str], limit: int = 10) -> List[Item]:
        """按分类筛选，按重要性排序"""
        with self._mu:
            cat_set = set(categories)
            filtered = [it for it in self._items if it.category in cat_set]
            filtered.sort(key=lambda x: x.importance, reverse=True)
            return filtered[:limit]

    def need_consolidation(self) -> bool:
        if not self._cfg:
            return False
        return self._store_count >= self._cfg.MemoryConsolidationTrigger

    def consolidate(self):
        """合并 / 衰减 / 过期淘汰"""
        with self._mu:
            if not self._cfg:
                return

            cfg = self._cfg
            now = time.time()

            # Phase 1: 重要性衰减
            for it in self._items:
                days = (now - it.created_at) / 86400
                it.importance *= cfg.MemoryConsolidationDecayRate ** days

            # Phase 2: 去重 + 合并
            removed_ids = set()
            merged = []
            n = len(self._items)
            for i in range(n):
                if i in removed_ids:
                    continue
                for j in range(i + 1, n):
                    if j in removed_ids:
                        continue
                    a, b = self._items[i], self._items[j]
                    if a.embedding and b.embedding:
                        sim = self._cosine(a.embedding, b.embedding)
                        if sim >= cfg.MemoryConsolidationDedup:
                            # 保留 importance 更高的
                            if a.importance >= b.importance:
                                removed_ids.add(j)
                            else:
                                removed_ids.add(i)
                                break
                        elif sim >= cfg.MemoryConsolidationSimilarity:
                            # 合并：保留较长内容
                            if len(b.content) > len(a.content):
                                a.content = b.content
                            a.importance = max(a.importance, b.importance)
                            removed_ids.add(j)

            # 删除被合并/去重的条目
            if removed_ids:
                self._items = [it for i, it in enumerate(self._items) if i not in removed_ids]

            # Phase 3: 过期淘汰
            ttl_seconds = cfg.MemoryConsolidationTTLDays * 86400
            self._items = [
                it for it in self._items
                if not ((now - it.created_at) > ttl_seconds and it.importance < cfg.MemoryConsolidationMinImport)
            ]

            # 重建 TF 词表
            self._rebuild_vocab()
            self._store_count = 0

    def count(self) -> int:
        with self._mu:
            return len(self._items)

    def snapshot(self) -> List[Item]:
        with self._mu:
            return list(self._items)

    def _rebuild_vocab(self):
        self._vocab = {}
        for it in self._items:
            self._build_tf_for(it)

    def _build_tf_for(self, item: Item):
        words = item.content.lower().split()
        if not words:
            return
        tf: Dict[str, float] = {}
        for w in words:
            if w not in self._vocab:
                self._vocab[w] = len(self._vocab)
        total = len(words)
        counts = Counter(words)
        for w, c in counts.items():
            tf[w] = c / total
        item.tf_vector = tf

    def _text_to_tf(self, text: str) -> Dict[str, float]:
        words = text.lower().split()
        total = len(words)
        if total == 0:
            return {}
        counts = Counter(words)
        return {w: c / total for w, c in counts.items()}

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _cosine_tf(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        all_keys = set(a.keys()) | set(b.keys())
        dot = sum(a.get(k, 0) * b.get(k, 0) for k in all_keys)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
