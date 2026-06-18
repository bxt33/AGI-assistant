"""ContextSource 抽象接口 + Query 数据结构

=============================================================================
                    📥 所有 Source 的基类和输入数据
=============================================================================

Query：每次 assemble 时的输入（告诉 Source"用户问了什么"）
ContextSource：所有 Source 必须实现的抽象接口（3 个方法）

继承关系：
  ContextSource
    ├── ProfileSource(source_profile.py)
    ├── PlannerSource(source_planner.py)
    ├── TaskMemSource(source_taskmem.py)
    ├── ToolStateSource(source_tools.py)
    ├── ConstraintsSource(source_constraints.py)
    └── RecallSource(source_recall.py)
=============================================================================
"""

from abc import ABC, abstractmethod
from typing import List
from src.domain.promptctx.slot import Slot, SlotKind, ContextItem
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Query:
    """
    每次 assemble 时的输入 — 告诉 Source"用户问了什么"。

    字段说明：
      text      — 用户原始查询文本（"北京天气怎么样？"）
      embedding — 查询的向量化表示（用于 LTM 语义召回）
      task_id   — 当前 ReAct 任务 ID（没有则为空字符串）
      mode      — 当前路由模式："chat" | "tool" | "rag" | "react"

    构造位置：ctx_builder.build_context_prefix()
      return agent._pctx.assemble(CtxQuery(
          text=query, embedding=emb, task_id=task_id, mode=mode,
      ))
    """
    text: str = ""
    embedding: Optional[List[float]] = None
    task_id: str = ""
    mode: str = "chat"  # chat / tool / react / rag


class ContextSource(ABC):
    """
    认知槽位的数据提供者 — 所有 Source 必须实现此接口。

    三个方法：
      id()        → 返回 Source 唯一标识（"profile" / "recall" / ...）
      supports()  → 返回是否支持某种 SlotKind（用于自动注册到 SourceRegistry）
      fetch()     → 根据 Slot 的过滤条件 + Query 内容，拉取数据

    注册方式（见 core_agent._build_prompt_ctx）：
      reg.register(ProfileSource(...))
      reg.register(RecallSource(...))
      ...
      → SourceRegistry 自动调用每个 Source 的 supports() 判断归属
      → 同一种 SlotKind 可以有多个 Source（但当前每种只有一个）
    """

    @abstractmethod
    def id(self) -> str:
        """返回 Source 唯一标识（用于日志和 meta.source 字段）"""
        ...

    @abstractmethod
    def supports(self, kind: SlotKind) -> bool:
        """判断此 Source 是否为给定 SlotKind 提供数据"""
        ...

    @abstractmethod
    def fetch(self, slot: Slot, q: Query) -> List[ContextItem]:
        """
        根据槽位定义和查询内容，拉取数据。

        参数：
          slot — 槽位定义（含 filter，告诉 Source 要什么、要多少）
          q    — 查询上下文（用户问题 + embedding + 模式）

        返回：
          [ContextItem, ...]  — 填充好的数据项列表
        """
        ...
