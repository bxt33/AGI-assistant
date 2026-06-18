"""Constraints 槽位 source：来自 sandbox 的静态安全策略

=============================================================================
                    🔒 告诉 LLM "你不能做什么"
=============================================================================

数据来源：sandbox/validator.policy_snapshot()（启动时一次性注册，不再变化）

安全策略分两级：
  BLOCK  — 绝对不能做（如 "禁止删除根路径"、"禁止修改系统文件"）
  WARN   — 做了可能有问题（如 "访问外部网络可能导致超时"）

渲染效果：
  【硬性约束】
  - [禁止] 禁止删除或修改 /etc 路径下的文件
  - [禁止] 禁止执行 rm -rf / 等危险命令
  - [告警] 网络访问可能被限制

适用模式：所有模式都包含此槽位
  CHAT_SCHEMA:  required=False（聊天时可有可无）
  REACT_SCHEMA: required=True （ReAct 时必须有——要执行命令了）
=============================================================================
"""

from typing import List
from src.domain.promptctx.source import ContextSource, Query
from src.domain.promptctx.slot import Slot, SlotKind, ContextItem
from src.domain.sandbox.validator import Policy, RiskLevel


class ConstraintsSource(ContextSource):
    """
    拉取沙箱安全策略作为 LLM 的行为约束。

    注入方式：
      reg.register(ConstraintsSource(policy_snapshot()))
      → policy_snapshot() 返回启动时的安全策略列表（全局静态）

    为什么先排 BLOCK 再排 WARN？
      最重要的约束放前面——全局裁剪时优先砍后面的告警，不砍禁止项。
    """

    def __init__(self, policies: List[Policy]):
        self._policies = list(policies)   # 拷贝一份，防止外部修改

    def id(self) -> str:
        return "constraints"

    def supports(self, kind: SlotKind) -> bool:
        return kind == SlotKind.CONSTRAINTS

    def fetch(self, slot: Slot, q: Query) -> List[ContextItem]:
        if not self._policies:
            return []

        # ── 排序：BLOCK 在前（优先级高），WARN 在后 ──
        blocks = [p for p in self._policies if p.level == RiskLevel.BLOCK]
        warns = [p for p in self._policies if p.level != RiskLevel.BLOCK]
        ordered = blocks + warns

        # 受 top_k 限制截断
        top_k = slot.filter.top_k
        if top_k > 0 and len(ordered) > top_k:
            ordered = ordered[:top_k]

        items = []
        for p in ordered:
            level = "禁止" if p.level == RiskLevel.BLOCK else "告警"
            score = 1.0 if p.level == RiskLevel.BLOCK else 0.5  # BLOCK 高分→不容裁剪
            items.append(ContextItem(
                text=f"[{level}] {p.reason}",
                score=score,
                source=self.id(),
                meta={"level": p.level.value, "pattern": p.pattern},
            ))
        return items
