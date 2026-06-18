"""Profile 槽位 source：用户偏好 + LTM 身份/偏好类条目

=============================================================================
                    👤 告诉 LLM"用户是谁"
=============================================================================

数据来源：
  ① Preference 存储（extract_preferences 提取的 key-value 对）
     例：{"姓名": "张三", "回答风格": "简洁", "技术栈": "Python"}

  ② LTM 中分类为 identity / preference 的长期记忆
     例："用户叫张三"、"用户喜欢简洁的回答"

渲染效果：
  【用户画像】
  - 姓名: 张三
  - 回答风格: 简洁
  - 技术栈: Python
  - 用户喜欢用 Python 做后端开发（重要性=0.62）

适用模式：所有模式（chat/tool/rag/react 都包含 PROFILE 槽位）
=============================================================================
"""

from typing import Optional, List
from src.domain.promptctx.source import ContextSource, Query
from src.domain.promptctx.slot import Slot, SlotKind, ContextItem
from src.domain.memory.preference import Preference
from src.domain.memory.longterm import LongTerm


class ProfileSource(ContextSource):
    """
    从 Preference 字典 + LTM 中提取用户画像信息。

    注入方式：
      reg.register(ProfileSource(self._mem.pref, self._mem.ltm))
    """

    def __init__(self, pref: Optional[Preference] = None,
                 ltm: Optional[LongTerm] = None):
        self._pref = pref      # Preference 存储（内存 dict）
        self._ltm = ltm        # 长期记忆（用于按分类过滤拉取身份信息）

    def id(self) -> str:
        return "profile"

    def supports(self, kind: SlotKind) -> bool:
        return kind == SlotKind.PROFILE

    def fetch(self, slot: Slot, q: Query) -> List[ContextItem]:
        """
        拉取用户画像数据。

        先取 Preference 的 key-value 对，再取 LTM 中分类匹配的条目。
        两部分来自不同存储，但放在同一个 "用户画像" 段落。
        """
        items: List[ContextItem] = []

        # ── ① 从 Preference 存储拉取 ──
        #     格式化为 "key: value" 形式（按 key 排序保证稳定）
        if self._pref:
            data = self._pref.snapshot()              # {"姓名": "张三", "回答风格": "简洁"}
            if data:
                for k in sorted(data.keys()):
                    items.append(ContextItem(
                        text=f"{k}: {data[k]}",        # "姓名: 张三"
                        score=1.0,                     # 偏好数据固定最高分
                        source=self.id(),
                    ))

        # ── ② 从 LTM 拉取分类匹配项 ──
        #     过滤 categories=["identity", "preference"]，取 top_k 条
        #     这些是 LLM 之前从对话中提取的身份类长期记忆
        if self._ltm and slot.filter.categories:
            limit = slot.filter.top_k if slot.filter.top_k > 0 else 10
            for item in self._ltm.filter_by_category(slot.filter.categories, limit):
                items.append(ContextItem(
                    text=item.content,
                    score=item.importance,             # 用 LTM importance 作为分数
                    source=self.id(),
                    meta={"category": item.category},
                ))

        return items
