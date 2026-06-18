"""文本分块器：滑动窗口 + 重叠切分

=============================================================================
                    ✂️ 把长文档切成小块
=============================================================================

位置（调用链）：
  engine.ingest(text)
    → split_text(text, chunk_size=200, chunk_overlap=50)
    → 返回 ["Python是一种解释型语言...", "广泛用于AI开发...", ...]

为什么需要切块？
  ① LLM 上下文窗口有限（不能一次喂 5000 字）
  ② 检索粒度：太粗（整篇文档）→ 匹配不准，太细（每句话）→ 丢失上下文
  ③ 200 字 + 50 字重叠是一个经过验证的平衡点

两种切法：
  split_text()        — 平面切块：等宽滑动窗口，边检索边返回
  split_with_parent() — Small-to-Big：小块检索精准，大块喂 LLM 保证完整性
=============================================================================
"""

from typing import List
from dataclasses import dataclass


@dataclass
class Chunk:
    """文档块数据结构

    字段说明：
      id             — 块编号（内存索引，从 0 自增）
      content        — 小块内容（给检索用，200 字）
      parent_content — 父块内容（给 LLM 看，1500 字，Small-to-Big 模式用）
    """
    id: int = 0
    content: str = ""
    parent_content: str = ""


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    滑动窗口切块：等宽 + 重叠。

    参数：
      text          — 原始文档文本
      chunk_size    — 每块最大字符数（默认 1000，项目配置为 200）
      chunk_overlap — 相邻块之间的重叠字符数（默认 200，项目配置为 50）

    返回：
      纯文本列表，如 ["Python是一种解释型语言...", "广泛用于AI开发...", ...]

    工作原理：
      text = "0123456789"（10字符），chunk_size=4，overlap=2

      第1块: [0 1 2 3]           → "0123"
      第2块:     [2 3 4 5]       → "2345"  （重叠 "23"）
      第3块:         [4 5 6 7]   → "4567"
      第4块:             [6 7 8 9] → "6789"

      为什么重叠？
        "Python 的异步编程使用 asyncio" 如果刚好在 200 字的边界被切断：
        → 块 A: "Python 的异步编程"    块 B: "使用 asyncio 库来处理"
        两个块单独看都不完整，overlap 确保关键句子不会被切断
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)    # 块结束位置
        chunk = text[start:end].strip()             # 取出当前块
        if chunk:
            chunks.append(chunk)
        if end >= text_len:                         # 到文本末尾，结束
            break
        start = end - chunk_overlap                 # 下一块起点 = 当前终点 - 重叠量
        if start <= 0 or start >= text_len:         # 防止死循环
            break

    return chunks


def split_with_parent(text: str, chunk_size: int = 500,
                      parent_size: int = 1500, overlap: int = 100) -> List[Chunk]:
    """
    Small-to-Big 双层分块：小块检索精准，大块返回给 LLM 保证上下文完整。

    为什么需要双层？
      embedding 向量检索时，小块匹配更精准（200 字比 1500 字更聚焦）
      但喂给 LLM 时，200 字可能丢失关键上下文（上文没看到，不知道在说什么）
      → 检索用小块（content），生成用大块（parent_content）

    结构示意：
      parent_size = 1500, chunk_size = 500, overlap = 100

      ┌──────────── parent 0（1500 字）────────────┐
      │ [chunk 0:500字] [chunk 1:500字] [chunk 2] │
      └────────────────────────────────────────────┘
      ┌──────────── parent 1（1500 字）────────────┐
      │          [chunk 3:500字] [chunk 4] [...] │
      └────────────────────────────────────────────┘

    返回：
      [
        Chunk(id=0, content="块0的500字", parent_content="父块0的1500字"),
        Chunk(id=1, content="块1的500字", parent_content="父块0的1500字"),
        Chunk(id=2, content="块2的500字", parent_content="父块0的1500字"),
        Chunk(id=3, content="块3的500字", parent_content="父块1的1500字"),
        ...
      ]

    注意：当前项目默认使用 split_text() 平面切块，split_with_parent 是预留的高级选项。
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)
    idx = 0

    while start < text_len:
        end = min(start + parent_size, text_len)
        parent = text[start:end].strip()            # 先切大块（parent）

        # 在大块内部切小块（child）
        sub_start = 0
        while sub_start < len(parent):
            sub_end = min(sub_start + chunk_size, len(parent))
            child = parent[sub_start:sub_end].strip()
            if child:
                # 小块内容给检索，大块内容给 LLM
                chunks.append(Chunk(id=idx, content=child, parent_content=parent))
                idx += 1
            if sub_end >= len(parent):
                break
            sub_start = sub_end - overlap            # 小块之间也有重叠

        if end >= text_len:
            break
        start = end - overlap                        # 大块之间也有重叠

    return chunks
