"""认知槽位类型定义

=============================================================================
                    🧩 提示词装配系统的数据结构
=============================================================================

整个 PromptCtx 系统围绕这几个核心类型运转：

  SlotKind    — 槽位类型枚举（PROFILE / PLANNER / TASK_MEMORY / ...）
  SlotFilter  — 过滤器（限定这个 Slot 要什么数据、不要什么）
  Slot        — 槽位定义（类型 + 是否必须 + 过滤条件）
  ContextItem — 槽位中的一条数据（Source 产出的最小单元）
  FilledSlot  — 填好的槽位（包含多条 ContextItem + 状态）

数据流：
  Schema 定义 [Slot, Slot, ...]（这套模式需要哪些槽位）
    → Source.fetch(Slot, Query) → [ContextItem, ...]（拉数据）
    → FilledSlot(kind, items, skipped?)（填充结果）
    → RuntimeContext.render()（渲染为中文提示词）
=============================================================================
"""

from enum import Enum
from typing import List, Dict
from dataclasses import dataclass, field


class SlotKind(str, Enum):
    """
    6 种槽位类型，对应 6 个 Source。

    PROFILE        — 用户画像（偏好 + 身份信息）
    PLANNER        — 任务规划（ReAct 模式下当前任务进度）
    TASK_MEMORY    — 任务记忆（ReAct 执行过程中每步的观察结果）
    TOOL_STATE     — 工具状态（可用工具列表 + 调用痕迹）
    CONSTRAINTS    — 硬性约束（沙箱安全策略）
    RECALL_MEMORY  — 相关回忆（LTM/GraphMemory 语义召回）
    """
    PROFILE = "profile"
    PLANNER = "planner"
    TASK_MEMORY = "task_memory"
    TOOL_STATE = "tool_state"
    CONSTRAINTS = "constraints"
    RECALL_MEMORY = "recall_memory"


@dataclass
class SlotFilter:
    """
    槽位过滤器 — 告诉 Source"我要什么样的数据"。

    每个字段的含义：
      categories    — 只取这些分类的记忆（如 ["identity", "preference"]）
      require_tags  — 必须有这些标签（如 ["python"]）
      min_score     — 最低相关性分数（低于此分丢弃）
      top_k         — 最多取几条（0 = 不限）
      max_age_hours — 只取 N 小时内的数据（0 = 不限）
      token_budget  — Token 预算上限（超出裁剪）
    """
    categories: List[str] = field(default_factory=list)
    require_tags: List[str] = field(default_factory=list)
    min_score: float = 0.4
    top_k: int = 0
    max_age_hours: int = 0
    token_budget: int = 0


@dataclass
class Slot:
    """
    槽位定义 — Schema 中的一个"格子"。

    例：TOOL_SCHEMA 中的 TOOL_STATE 槽位：
      Slot(
        kind=SlotKind.TOOL_STATE,
        required=True,              # ReAct 模式下必须有工具信息
        filter=SlotFilter(
          token_budget=350,         # 最多 350 tokens
          top_k=6                   # 最多 6 条
        )
      )

    required=True → 即使 Source 返回空，也会标记 skipped=False（不裁剪）
    required=False → Source 返回空可以跳过，不报错
    """
    kind: SlotKind
    required: bool = False
    filter: SlotFilter = field(default_factory=SlotFilter)
    template: str = ""  # 渲染模板（当前未使用，保留给未来自定义格式）


@dataclass
class ContextItem:
    """
    Source.fetch() 产出的最小数据单元 — 槽位中的"一条"信息。

    例：
      ContextItem(
        text="姓名: 张三",
        score=1.0,
        source="profile",
        meta={"category": "identity"}
      )

    渲染后变成：【用户画像】 的一行 "- 姓名: 张三"
    """
    text: str = ""
    score: float = 0.0       # 相关性/重要性分数（裁剪时低分先砍）
    source: str = ""         # 数据来源标识（"profile" / "recall" / ...）
    meta: Dict[str, str] = field(default_factory=dict)  # 附加元数据


@dataclass
class FilledSlot:
    """
    填好的槽位 — ContextAssembler 的产出单元。

    三种状态：
      ① items 有数据，skipped=False  → 正常填充
      ② items 为空，skipped=True     → 没有 Source 提供数据（或 Source 返回空）
      ③ items 被清空，skipped=True   → 全局预算超限后被裁剪

    reason 记录为什么跳过：
      "no source registered"  — 这个 SlotKind 没有注册 Source
      "source returned empty" — Source 有但没返回数据
      "global budget exceeded"— 被全局预算裁剪了
    """
    kind: SlotKind
    items: List[ContextItem] = field(default_factory=list)
    skipped: bool = False
    reason: str = ""
