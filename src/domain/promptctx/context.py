"""RuntimeContext 渲染：把填好的槽位变成中文提示词

=============================================================================
                    🖨️ 最后一步：数据 → 文字
=============================================================================

位置（调用链）：
  ContextAssembler.assemble()
    → 选 Schema → 并发填 Slot → 收集 FilledSlot → 预算裁剪
    → 返回 RuntimeContext(schema, filled)
    → rc.render()   ← 就是这里
    → 返回中文提示词字符串

渲染入口：RuntimeContext.render()
  遍历 filled 列表 → 跳过空槽位 → 每个非空槽位调 _render_slot()

单槽位渲染：_render_slot(fs)
  【槽位标题】
  - 内容项1
  - 内容项2

最终产物示例：
  【硬性约束】
  - [禁止] 禁止删除根路径

  【用户画像】
  - 姓名: 张三
  - 回答风格: 简洁

  【相关回忆】
  - 用户问了Python异步问题（重要性=0.62, 综合分=0.78）
=============================================================================
"""

from typing import List, Optional
from src.domain.promptctx.slot import FilledSlot, SlotKind, ContextItem
from src.domain.promptctx.schema import RuntimeContextSchema


class RuntimeContext:
    """
    ContextAssembler 的最终产出 → 包含 Schema 和填充好的槽位。

    两个主要用途：
      ① render() → 生成中文提示词前缀给 LLM
      ② slot_by_kind() → 按类型查找特定槽位（程序化读取）
    """

    def __init__(self, schema: RuntimeContextSchema, filled: List[FilledSlot]):
        self.schema = schema
        self.filled = filled      # 按 Schema 中 Slot 的顺序排列
        self.trace: List[str] = []  # 调试追踪（当前未使用，预留给未来日志）

    def slot_by_kind(self, kind: SlotKind) -> Optional[FilledSlot]:
        """按类型查找已填充的槽位（程序化读取）"""
        for fs in self.filled:
            if fs.kind == kind:
                return fs
        return None

    def render(self) -> str:
        """
        将所有非空槽位按 Schema 顺序渲染为中文提示前缀。

        渲染规则：
          - 跳过的槽位（skipped=True）不渲染
          - 空 items 的槽位不渲染
          - 每个槽位生成一个 "【标题】\n- item1\n- item2" 的段落
          - 段落之间用 "\n\n" 分隔

        返回：
          完整的中文提示词前缀字符串，如：
          【硬性约束】\n- [禁止] 禁止删除根路径\n\n【用户画像】\n- 姓名: 张三
        """
        if not self.filled:
            return ""

        sections = []
        for fs in self.filled:
            if fs.skipped or not fs.items:
                continue
            sections.append(_render_slot(fs))
        return "\n\n".join(sections)


# ── 槽位类型 → 中文标题映射 ──
_SLOT_TITLES = {
    SlotKind.PROFILE: "用户画像",
    SlotKind.PLANNER: "任务规划",
    SlotKind.TASK_MEMORY: "任务记忆",
    SlotKind.TOOL_STATE: "可用工具",
    SlotKind.CONSTRAINTS: "硬性约束",
    SlotKind.RECALL_MEMORY: "相关回忆",
}


def _render_slot(fs: FilledSlot) -> str:
    """
    渲染单个已填充的槽位。

    格式：
      【用户画像】
      - 姓名: 张三
      - 回答风格: 简洁

    每行前面加 "- " 前缀，过滤空文本。
    """
    title = _SLOT_TITLES.get(fs.kind, fs.kind.value)   # 取中文标题
    lines = [f"- {item.text}" for item in fs.items if item.text.strip()]
    if not lines:
        return ""
    return f"【{title}】\n" + "\n".join(lines)
