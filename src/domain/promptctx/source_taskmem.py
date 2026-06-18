"""Task Memory 槽位 source：ReAct 步骤观察缓冲区

=============================================================================
                    📝 告诉 LLM "前几步执行了什么结果"
=============================================================================

数据来源：TaskMemBuffer（环形缓冲区，最多保留 20 条步骤观察）

在 ReAct 执行过程中：
  每完成一步 → push_task_mem(StepObservation) → 写入 buffer
  下一步的 LLM 调用 → fetch → 读取 buffer → 拼到提示词里

渲染效果：
  【任务记忆】
  - 步骤0 [get_weather]→北京 晴 22°C
  - 步骤1 [search_web] 失败: API超时

TaskMemBuffer vs TaskRuntime：
  TaskRuntime — 追踪任务结构（几步、每步状态、进度）
  TaskMemBuffer — 记录每步的详细结果（工具输出、错误信息）

两个是互补的：一个管"进度条"，一个管"草稿纸"。
=============================================================================
"""

import threading
import time
from typing import List, Optional
from dataclasses import dataclass, field
from src.domain.promptctx.source import ContextSource, Query
from src.domain.promptctx.slot import Slot, SlotKind, ContextItem


@dataclass
class StepObservation:
    """
    单步执行的观察结果。

    在 core_agent._run_react() 中每完成一步后写入：
      self._pctx.push_task_mem(StepObservation(
          step_id=i, tool_name=tool.name, result=result,
          error=error, success=True/False
      ))
    """
    step_id: int = 0
    tool_name: str = ""
    result: str = ""            # 工具执行结果（成功时）
    error: str = ""             # 错误信息（失败时）
    success: bool = True
    created_at: float = 0.0     # 记录时间戳


class TaskMemBuffer:
    """
    环形缓冲区 — 保存最近 N 条步骤观察。

    为什么是环形的？
      ReAct 可能执行很多步（10步、20步），但上下文窗口有限。
      只保留最近 20 步，旧的自动淘汰——足够恢复上下文，又不会爆 tokens。

    线程安全：用 RLock 保护读写，因为写入在 Executor 线程，读取在 assemble 线程。
    """

    def __init__(self, max_size: int = 20):
        self._mu = threading.RLock()
        self._buf: List[StepObservation] = []
        self._max = max(1, max_size)

    def push(self, obs: StepObservation):
        """写入一条观察结果（超过容量自动淘汰最旧的）"""
        with self._mu:
            if obs.created_at == 0:
                obs.created_at = time.time()
            self._buf.append(obs)
            if len(self._buf) > self._max:
                self._buf = self._buf[-self._max:]   # 保留最后 max_size 条

    def reset(self):
        """清空缓冲区（新任务开始时调用，换一张"草稿纸"）"""
        with self._mu:
            self._buf.clear()

    def snapshot(self) -> List[StepObservation]:
        """返回当前所有观察的副本（线程安全）"""
        with self._mu:
            return list(self._buf)


class TaskMemSource(ContextSource):
    """
    从 TaskMemBuffer 读取最近步骤的观察结果。

    注入方式：
      reg.register(TaskMemSource(self._pctx.task_mem))
      → self._pctx = PromptCtx() → task_mem = TaskMemBuffer(20)
    """

    def __init__(self, buf: Optional[TaskMemBuffer] = None):
        self._buf = buf

    def id(self) -> str:
        return "task_memory"

    def supports(self, kind: SlotKind) -> bool:
        return kind == SlotKind.TASK_MEMORY

    def fetch(self, slot: Slot, q: Query) -> List[ContextItem]:
        """
        读取步骤观察记录。

        只返回最近 top_k 条（由 schema 中的 filter.top_k 控制，默认 8）。
        每条结果截断到 200 字符——够了解"发生了什么"，不浪费 tokens。
        """
        if not self._buf:
            return []

        obs_list = self._buf.snapshot()
        if not obs_list:
            return []

        # 只取最近 N 条（ReAct 很多步时控制 tokens）
        top_k = slot.filter.top_k
        if top_k > 0 and len(obs_list) > top_k:
            obs_list = obs_list[-top_k:]   # 最新的优先级更高

        items = []
        for o in obs_list:
            text = f"步骤{o.step_id} [{o.tool_name}]"
            if o.success:
                r = o.result
                if len(r) > 200:
                    r = r[:200] + "…"       # 截断长结果
                text += "→" + r
            else:
                text += f" 失败: {o.error}"
            items.append(ContextItem(
                text=text,
                source=self.id(),
                meta={"tool": o.tool_name},
            ))
        return items
