"""工具注册表：并发安全的工具集管理"""

import threading
from typing import Dict, Optional, List
from src.domain.tool import Tool


class ToolRegistry:
    def __init__(self, initial: Optional[Dict[str, Tool]] = None):
        self._mu = threading.RLock()
        self._tools: Dict[str, Tool] = dict(initial) if initial else {}

    def register(self, tool: Tool):
        with self._mu:
            self._tools[tool.name] = tool

    def filter(self, names: List[str]) -> Dict[str, Tool]:
        with self._mu:
            name_set = set(names)
            return {k: v for k, v in self._tools.items() if k in name_set}

    def snapshot(self) -> Dict[str, Tool]:
        with self._mu:
            return dict(self._tools)
