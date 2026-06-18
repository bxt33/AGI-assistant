"""短期记忆：固定窗口的对话历史。
维护最近 MaxTurns 轮（每轮 user + assistant 两条）的对话上下文，
超出窗口时自动丢弃最早记录。不持久化——进程消亡即清空。
"""

import threading
import time
from typing import List
from dataclasses import dataclass, field


@dataclass
class ConversationMessage:
    role: str
    content: str
    timestamp: str = ""


class ShortTerm:
    """维护最近 MaxTurns 轮的对话上下文"""

    def __init__(self, max_turns: int = 10):
        self._mu = threading.RLock()
        self._messages: List[ConversationMessage] = []
        self.max_turns = max_turns

    def add(self, role: str, content: str):
        with self._mu:
            ts = time.strftime("%H:%M:%S")
            self._messages.append(ConversationMessage(
                role=role, content=content, timestamp=ts
            ))
            max_msgs = self.max_turns * 2  # 每轮 = user + assistant
            if len(self._messages) > max_msgs:
                self._messages = self._messages[-max_msgs:]

    def snapshot(self) -> List[ConversationMessage]:
        with self._mu:
            return list(self._messages)

    def count(self) -> int:
        with self._mu:
            return len(self._messages)
