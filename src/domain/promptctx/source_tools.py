"""Tool State 槽位 source：可用工具清单 + 近期调用记录

=============================================================================
                    🔧 告诉 LLM "我能用什么工具、之前用了什么"
=============================================================================

数据来源：
  ① ToolRegistry.snapshot()（通过 lambda 回调实时读取工具列表）
  ② ToolStateTracker（环形缓冲区，记录最近 10 次工具调用）

渲染效果：
  【可用工具】
  - get_time — 获取当前时间（必填 timezone）
  - get_weather — 查询指定城市的天气（必填 city）
  - rag_search — 从私人黑洞中检索相关文档内容
  - 近期调用 get_weather [成功]: 北京 晴 22°C

适用模式：
  TOOL_SCHEMA、REACT_SCHEMA 包含 TOOL_STATE 槽位
  Chat 和 RAG 模式没有此槽位
=============================================================================
"""

import threading
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

from src.domain.promptctx.source import ContextSource, Query
from src.domain.promptctx.slot import Slot, SlotKind, ContextItem
from src.domain.tool import Tool


@dataclass
class ToolCallTrace:
    """
    单次工具调用的痕迹记录。

    每调用一次工具（无论是 Tool 模式还是 ReAct 模式），都会记录一条。
    记录在 core_agent 的 _run_tool() 和 _run_react() 中写入。
    """
    tool_name: str = ""
    success: bool = True        # 调用是否成功
    summary: str = ""           # 结果摘要（最多 120 字符）
    created_at: float = 0.0     # 调用时间


class ToolStateTracker:
    """
    工具调用痕迹的环形缓冲区 — 保留最近 N 条调用记录。

    区别于 TaskMemBuffer：
      TaskMemBuffer  — ReAct 每步详细结果（200 字符），只保留当前任务
      ToolStateTracker — 所有模式的调用痕迹（120 字符摘要），跨任务保留
    """

    def __init__(self, max_size: int = 10):
        self._mu = threading.RLock()
        self._buf: List[ToolCallTrace] = []
        self._max = max(1, max_size)

    def record(self, trace: ToolCallTrace):
        """记录一次工具调用"""
        with self._mu:
            if trace.created_at == 0:
                trace.created_at = time.time()
            if len(trace.summary) > 120:
                trace.summary = trace.summary[:120] + "…"  # 截断摘要
            self._buf.append(trace)
            if len(self._buf) > self._max:
                self._buf = self._buf[-self._max:]          # 环形淘汰

    def snapshot(self) -> List[ToolCallTrace]:
        """返回当前所有痕迹的副本"""
        with self._mu:
            return list(self._buf)


# 工具注册表回调类型：无参数，返回 {name: Tool}
ToolRegistryProvider = Callable[[], Dict[str, Tool]]


class ToolStateSource(ContextSource):
    """
    拉取可用工具列表 + 近期调用痕迹。

    注入方式：
      reg.register(ToolStateSource(
          lambda: self._tools.snapshot(),   # 每次实时读取工具列表
          self._pctx.tool_tracker            # 工具调用痕迹缓冲区
      ))
    """

    def __init__(self, registry: Optional[ToolRegistryProvider] = None,
                 tracker: Optional[ToolStateTracker] = None):
        self._registry = registry
        self._tracker = tracker

    def id(self) -> str:
        return "tool_state"

    def supports(self, kind: SlotKind) -> bool:
        return kind == SlotKind.TOOL_STATE

    def fetch(self, slot: Slot, q: Query) -> List[ContextItem]:
        items: List[ContextItem] = []

        # ── ① 可用工具清单 ──
        #     格式："工具名 — 描述（必填 参数1, 参数2）"
        #     按名称排序保证输出稳定
        if self._registry:
            tool_map = self._registry()
            for name in sorted(tool_map.keys()):
                t = tool_map[name]
                param_hint = ""
                for p in t.parameters:
                    if p.required:
                        if param_hint:
                            param_hint += ", "
                        param_hint += p.name
                if param_hint:
                    param_hint = f"（必填 {param_hint}）"
                items.append(ContextItem(
                    text=f"{name} — {t.description}{param_hint}",
                    source=self.id(),
                    meta={"tool": name},
                ))

        # ── ② 近期调用痕迹 ──
        #     让 LLM 知道"刚才试过什么、结果如何"
        #     避免重复调用同一个失败的工具
        if self._tracker:
            traces = self._tracker.snapshot()
            top_k = slot.filter.top_k
            if top_k > 0 and len(traces) > top_k:
                traces = traces[-top_k:]   # 最近的最有价值
            for tr in traces:
                status = "成功" if tr.success else "失败"
                items.append(ContextItem(
                    text=f"近期调用 {tr.tool_name} [{status}]: {tr.summary}",
                    source=self.id(),
                    meta={"tool": tr.tool_name, "status": status},
                ))

        return items
