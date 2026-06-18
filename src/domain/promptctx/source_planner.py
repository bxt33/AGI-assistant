"""Planner 槽位 source：当前 ReAct 任务的规划状态

=============================================================================
                    📋 告诉 LLM"我在做什么、做到哪了"
=============================================================================

数据来源：TaskRuntime.current_task()（通过 lambda 回调实时读取）

只在 ReAct 模式下生效：REACT_SCHEMA 包含 PLANNER 槽位（required=True），
其他模式没有此槽位，不会调用 PlannerSource。

渲染效果：
  【任务规划】
  - 任务 task_1718123456789 状态=running 阶段=executing
  - 进度：第 2/3 步
  - 下一步：生成总结（工具=rag_search）

PlannerSnapshot 字段速查：
  task_id       → "task_1718123456789"
  query         → 用户原始问题
  status        → "running" / "completed" / "interrupted"
  phase         → "planning" / "executing" / "generating" / "done"
  total_steps   → 3
  current_step  → 1（从 0 开始，表示正在执行第 2 步）
  next_step_name→ "生成总结"
  next_step_tool→ "rag_search"
  interrupted_at→ 0（0=未中断，>0=被中断时的步骤索引）
=============================================================================
"""

from typing import Optional, Callable, List
from dataclasses import dataclass
from src.domain.promptctx.source import ContextSource, Query
from src.domain.promptctx.slot import Slot, SlotKind, ContextItem


@dataclass
class PlannerSnapshot:
    """
    ReAct 任务当前状态的快照。

    由 core_agent._planner_snapshot() 实时构建，通过 lambda 回调传给 Source。
    不是持久化数据，每次 assemble() 都重新读取。
    """
    task_id: str = ""           # 任务唯一 ID
    query: str = ""             # 用户原始问题
    status: str = ""            # running / completed / interrupted
    phase: str = ""             # planning / executing / generating / done / interrupted
    total_steps: int = 0        # 总步骤数
    current_step: int = 0       # 当前步骤索引（从 0 开始）
    interrupted_at: int = 0     # 被中断时的步骤索引（0=未中断）
    next_step_name: str = ""    # 下一步名称（原因描述）
    next_step_tool: str = ""    # 下一步使用的工具名


# 回调函数类型：无参数，返回当前任务的快照（或 None）
PlannerProvider = Callable[[], Optional[PlannerSnapshot]]


class PlannerSource(ContextSource):
    """
    从 TaskRuntime 读取当前 ReAct 任务进度。

    注入方式：
      reg.register(PlannerSource(lambda: self._planner_snapshot()))
      → 每次 fetch 时调 lambda 获取最新快照，不缓存
    """

    def __init__(self, provider: PlannerProvider):
        self._get = provider    # lambda: self._planner_snapshot()

    def id(self) -> str:
        return "planner"

    def supports(self, kind: SlotKind) -> bool:
        return kind == SlotKind.PLANNER

    def fetch(self, slot: Slot, q: Query) -> List[ContextItem]:
        """
        拉取当前任务规划状态。

        没有当前任务（非 ReAct 模式 或 任务已完成）→ 返回空列表。
        """
        if not self._get:
            return []

        snap = self._get()      # 实时读取，不缓存
        if not snap:
            return []

        # 拼装多条信息，每条作为一个 ContextItem
        items = [
            ContextItem(
                text=f"任务 {snap.task_id} 状态={snap.status} 阶段={snap.phase}",
                source=self.id(),
            )
        ]
        # 有步骤信息时追加进度
        if snap.total_steps > 0:
            items.append(ContextItem(
                text=f"进度：第 {snap.current_step + 1}/{snap.total_steps} 步",
                source=self.id(),
            ))
        # 有下一步时追加预告（让 LLM 提前知道接下来要做什么）
        if snap.next_step_name:
            items.append(ContextItem(
                text=f"下一步：{snap.next_step_name}（工具={snap.next_step_tool}）",
                source=self.id(),
            ))
        # 被中断过 → 告诉 LLM 可以从断点恢复
        if snap.status == "interrupted" and snap.interrupted_at > 0:
            items.append(ContextItem(
                text=f"上次在第 {snap.interrupted_at + 1} 步被中断，可从此处恢复",
                source=self.id(),
            ))
        return items
