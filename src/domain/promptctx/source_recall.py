"""Recall 槽位 source：LTM / GraphMemory 语义召回

=============================================================================
                    💭 告诉 LLM "你和用户之前聊过什么"
=============================================================================

数据来源：
  LongTerm.recall_by_filter()  — embedding 语义召回 + TF 词袋降级
  GraphMemory.recall_by_filter() — LTM 召回 + Neo4j 图扩展

二选一（core_agent._build_prompt_ctx 中决定）：
  图记忆可用 → RecallSource(self._mem.graph_mem)
  图不可用   → RecallSource(self._mem.ltm)

渲染效果：
  【相关回忆】
  - 用户之前问了Python异步编程问题（重要性=0.62, 综合分=0.78）
  - 用户常用 Python 做后端开发（重要性=0.55, 综合分=0.71）

适用模式：所有模式都包含 RECALL_MEMORY 槽位
  但不同模式的 filter 不同：
    CHAT_SCHEMA: top_k=3, min_score=0.4, categories=["episodic", "fact", "general"]
    REACT_SCHEMA: top_k=2, min_score=0.5, categories=[..., "tool_failure"]
=============================================================================
"""

from typing import Optional, List
from src.domain.promptctx.source import ContextSource, Query
from src.domain.promptctx.slot import Slot, SlotKind, ContextItem
from src.domain.memory.longterm import LongTerm, RecallFilter, Item


class Recaller:
    """
    LTM / GraphMemory 的统一抽象接口。

    设计意图：
      LTM 和 GraphMemory 都实现了 recall_by_filter()，
      通过这个抽象类，RecallSource 不需要知道底层是哪种实现。
      （Python 鸭子类型实际上不需要这个抽象类，
       但显式定义接口有助于理解架构意图）
    """
    def recall_by_filter(self, query: str, query_embedding: Optional[List[float]],
                         flt: RecallFilter) -> List[Item]:
        raise NotImplementedError


class RecallSource(ContextSource):
    """
    从 LTM 或 GraphMemory 做语义召回 + 图扩展。

    注入方式：
      # 图可用 → 图增强召回（LTM 基础召回 + Neo4j 图扩展）
      RecallSource(self._mem.graph_mem)
      # 图不可用 → 纯 LTM 语义召回
      RecallSource(self._mem.ltm)

    fetch() 流程：
      ① 用 Slot 的 filter 构造 RecallFilter（分类、标签、最低分数、最大数量）
      ② 调 recaller.recall_by_filter(query_text, query_embedding, flt)
      ③ 格式化输出："记忆内容（重要性=0.62, 综合分=0.78）"
    """

    def __init__(self, recaller: Optional[Recaller] = None):
        self._recaller = recaller

    def id(self) -> str:
        return "recall"

    def supports(self, kind: SlotKind) -> bool:
        return kind == SlotKind.RECALL_MEMORY

    def fetch(self, slot: Slot, q: Query) -> List[ContextItem]:
        if not self._recaller:
            return []

        # ── ① 用 Slot 的 filter 构造 RecallFilter ──
        #     Schema 中定义了什么类别的记忆需要召回（如 "episodic", "fact"）
        flt = RecallFilter(
            categories=slot.filter.categories,
            require_tags=slot.filter.require_tags,
            min_score=slot.filter.min_score,
            top_k=slot.filter.top_k,
            max_age_hours=slot.filter.max_age_hours,
        )

        # ── ② 语义召回 ──
        #     LTM：embedding 余弦相似度 + TF 词袋降级
        #     GraphMemory：LTM 召回 + Neo4j 图扩展（沿 FOLLOWS/SIMILAR_TO 多跳）
        hits = self._recaller.recall_by_filter(q.text, q.embedding, flt)
        if not hits:
            return []

        # ── ③ 格式化输出 ──
        #     每条记忆带分数，LLM 可以根据分数判断可靠性
        items = []
        for h in hits:
            meta = {}
            if h.category:
                meta["category"] = h.category
            if h.slot_hint:
                meta["slot_hint"] = h.slot_hint
            items.append(ContextItem(
                text=f"{h.content}（重要性={h.importance:.2f}, 综合分={h.score:.2f}）",
                score=h.score,
                source=self.id(),
                meta=meta,
            ))
        return items
