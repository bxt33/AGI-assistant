"""短期记忆：固定窗口的对话历史。

=============================================================================
                    🧠 记忆系统 第一层：短期记忆（STM）
=============================================================================

三层记忆架构：
  ┌─────────────┐  ← 第一层（这里）
  │  ShortTerm  │     滑动窗口对话历史，纯 Python 内存列表
  ├─────────────┤  ← 第二层
  │  LongTerm   │     语义向量召回 + TF 词袋降级 + 三阶段合并淘汰
  ├─────────────┤  ← 第三层
  │  GraphMemory│     Neo4j 图结构增强：沿 FOLLOWS/SIMILAR_TO 扩展召回
  ├─────────────┤  ← 独立维度
  │  Preference │     用户画像 KV（LLM提取 + 规则兜底）

ShortTerm 的特点：
  ✅ 容量固定：max_turns × 2 条（每轮 = user + assistant 各一条）
  ✅ 滑动窗口：超出自动丢弃最早的（FIFO）
  ❌ 不持久化：进程消亡即清空——"用完即忘"
  ❌ 不向量化：不做 embedding，只在 Python 内存中

为什么 STM 不持久化？
  短期记忆的价值是"当前会话的上下文连贯性"，跨进程就失效了。
  持久化是 LTM 的职责——LTM 会把重要内容向量化存入 PG/Milvus/ES。

在提示词中的位置：
  不通过 Schema/Source 体系注入，而是直接在 ctx_builder.build_history_messages()
  中从 agent.stm.snapshot() 读取，拼到 messages 列表里作为对话历史。

=============================================================================
"""

import threading
import time
from typing import List
from dataclasses import dataclass, field


@dataclass
class ConversationMessage:
    """
    单条对话消息。

    与 OpenAI Chat API 的 message 格式对齐：
      role: "user" | "assistant" | "system" | "tool"
      content: 消息正文
      timestamp: "HH:MM:SS" 格式，便于前端展示时间线
    """
    role: str
    content: str
    timestamp: str = ""


class ShortTerm:
    """
    维护最近 MaxTurns 轮的对话上下文。

    ── 数据结构 ──
    self._messages: List[ConversationMessage]
      一个简单的 Python 列表，没有持久化，没有向量化。
      每轮对话追加 user + assistant 两条。

    ── 容量控制 ──
    max_turns = 10 → 最多保留 20 条消息。
    超出后截断保留最后 20 条（丢弃最早的）。

    ── 线程安全 ──
    threading.RLock(): 可重入锁。
    因为 add() 和 snapshot() 可能在不同线程被调用（Executor 线程 vs HTTP 线程）。

    ── 与其他记忆的关系 ──
    ShortTerm: "刚才说了什么"          → 最近 N 轮原话
    LongTerm:  "以前聊过什么（语义）"  → embedding 召回相似内容
    Preference: "用户是谁"             → 姓名/风格/偏好 KV
    """

    def __init__(self, max_turns: int = 10):
        """
        Args:
            max_turns: 最大保留轮数。默认 10 轮 = 20 条消息。
                       每轮包含 user 一条 + assistant 一条。
                       设为 0 表示不保留任何历史（不推荐）。
        """
        self._mu = threading.RLock()
        self._messages: List[ConversationMessage] = []
        self.max_turns = max_turns

    def add(self, role: str, content: str):
        """
        追加一条消息并自动截断超出的部分。

        调用时机：
          core_agent._run_chat()   → stm.add("user", query)
          core_agent._run_chat()   → stm.add("assistant", final_answer)
          core_agent._run_tool()   → stm.add("user"/"assistant", ...)
          core_agent._run_react()  → 每步思考和行动都 add

        容量控制逻辑：
          max_msgs = max_turns × 2  （每轮 = user + assistant）
          超出后取 [-max_msgs:] → 丢弃最早的，保留最新的。

        示例：
          max_turns=3, 当前 6 条消息 (3轮)
          add("user", "新问题") → 7 条 → 截断为最后 6 条（丢最早 1 条）
        """
        with self._mu:
            ts = time.strftime("%H:%M:%S")
            self._messages.append(ConversationMessage(
                role=role, content=content, timestamp=ts
            ))
            max_msgs = self.max_turns * 2  # 每轮 = user + assistant
            if len(self._messages) > max_msgs:
                self._messages = self._messages[-max_msgs:]

    def snapshot(self) -> List[ConversationMessage]:
        """
        返回当前所有消息的副本（线程安全）。

        调用时机：
          ctx_builder.build_history_messages() → 拼接到 LLM messages 列表
          前端 API 获取对话历史 → 渲染聊天界面

        为什么返回副本而不是引用？
          防止外部代码在遍历时修改列表（线程安全）。
        """
        with self._mu:
            return list(self._messages)

    def count(self) -> int:
        """返回当前消息数量（用于调试和日志）"""
        with self._mu:
            return len(self._messages)
