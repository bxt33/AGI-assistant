"""记忆栈：聚合 短期记忆 / 长期记忆 / 图记忆 / 偏好"""

from src.domain.memory.shortterm import ShortTerm
from src.domain.memory.longterm import LongTerm
from src.domain.memory.preference import Preference
from src.domain.memory.graphmem import GraphMemory


class MemoryStack:
    def __init__(self, cfg):
        self.stm = ShortTerm(cfg.ShortTermMaxTurns)
        self.ltm = LongTerm(cfg)
        self.pref = Preference()
        self.graph_mem: GraphMemory = None  # 启动期由 initKnowledgeGraph 注入
