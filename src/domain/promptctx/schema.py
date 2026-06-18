"""Schema-driven 认知槽位编排"""

from typing import Dict, List
from dataclasses import dataclass, field
from src.domain.promptctx.slot import Slot, SlotKind, SlotFilter

DEFAULT_GLOBAL_TOKEN_BUDGET = 2400


@dataclass
class RuntimeContextSchema:
    mode: str = "chat"
    slots: List[Slot] = field(default_factory=list)


# ── Chat Schema ──
CHAT_SCHEMA = RuntimeContextSchema(
    mode="chat",
    slots=[
        Slot(kind=SlotKind.CONSTRAINTS, required=False,
             filter=SlotFilter(token_budget=200)),
        Slot(kind=SlotKind.PROFILE, required=False,
             filter=SlotFilter(categories=["identity", "preference"],
                               token_budget=300, top_k=10)),
        Slot(kind=SlotKind.RECALL_MEMORY, required=False,
             filter=SlotFilter(categories=["episodic", "fact", "general"],
                               top_k=3, min_score=0.4, token_budget=400)),
    ],
)

# ── Tool Schema ──
TOOL_SCHEMA = RuntimeContextSchema(
    mode="tool",
    slots=[
        Slot(kind=SlotKind.CONSTRAINTS, required=False,
             filter=SlotFilter(token_budget=200)),
        Slot(kind=SlotKind.PROFILE, required=False,
             filter=SlotFilter(categories=["identity", "preference"],
                               token_budget=250, top_k=8)),
        Slot(kind=SlotKind.TOOL_STATE, required=True,
             filter=SlotFilter(token_budget=350, top_k=6)),
        Slot(kind=SlotKind.RECALL_MEMORY, required=False,
             filter=SlotFilter(categories=["episodic", "fact", "general"],
                               top_k=2, min_score=0.5, token_budget=250)),
    ],
)

# ── React Schema ──
REACT_SCHEMA = RuntimeContextSchema(
    mode="react",
    slots=[
        Slot(kind=SlotKind.CONSTRAINTS, required=True,
             filter=SlotFilter(token_budget=280)),
        Slot(kind=SlotKind.PLANNER, required=True,
             filter=SlotFilter(token_budget=300)),
        Slot(kind=SlotKind.TASK_MEMORY, required=False,
             filter=SlotFilter(token_budget=350, top_k=8, max_age_hours=24)),
        Slot(kind=SlotKind.TOOL_STATE, required=True,
             filter=SlotFilter(token_budget=350, top_k=8)),
        Slot(kind=SlotKind.PROFILE, required=False,
             filter=SlotFilter(categories=["identity", "preference"],
                               token_budget=250, top_k=6)),
        Slot(kind=SlotKind.RECALL_MEMORY, required=False,
             filter=SlotFilter(categories=["episodic", "fact", "general", "tool_failure"],
                               top_k=2, min_score=0.5, token_budget=200)),
    ],
)

# ── RAG Schema ──
RAG_SCHEMA = RuntimeContextSchema(
    mode="rag",
    slots=[
        Slot(kind=SlotKind.CONSTRAINTS, required=False,
             filter=SlotFilter(token_budget=200)),
        Slot(kind=SlotKind.PROFILE, required=False,
             filter=SlotFilter(categories=["identity", "preference"],
                               token_budget=300, top_k=8)),
        Slot(kind=SlotKind.RECALL_MEMORY, required=False,
             filter=SlotFilter(categories=["episodic", "fact", "general"],
                               top_k=3, min_score=0.4, token_budget=400)),
    ],
)


def default_schemas() -> Dict[str, RuntimeContextSchema]:
    return {
        "chat": CHAT_SCHEMA,
        "tool": TOOL_SCHEMA,
        "react": REACT_SCHEMA,
        "rag": RAG_SCHEMA,
    }


def slot_priority(kind: SlotKind) -> int:
    """决定全局预算超限时的裁剪优先级（数字越小越优先保留）"""
    return {
        SlotKind.CONSTRAINTS: 0,
        SlotKind.PLANNER: 1,
        SlotKind.TASK_MEMORY: 2,
        SlotKind.TOOL_STATE: 3,
        SlotKind.PROFILE: 4,
        SlotKind.RECALL_MEMORY: 5,
    }.get(kind, 99)
