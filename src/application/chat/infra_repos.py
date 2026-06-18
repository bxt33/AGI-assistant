"""数据访问层 + 事件总线的聚合容器"""

from src.infrastructure.eventbus import Publisher, LogPublisher
from src.infrastructure.persistence.chathistory import Repo as ChatRepo
from src.infrastructure.persistence.preference import Repo as PrefRepo
from src.infrastructure.persistence.snapshot import Repo as SnapRepo
from src.infrastructure.persistence.longterm import Repo as LTMRepo
from src.infrastructure.persistence.ragchunk import Repo as RAGChunkRepo


class RepoBundle:
    def __init__(self, chat: ChatRepo, pref: PrefRepo, snap: SnapRepo,
                 ltm: LTMRepo, rag_chunk: RAGChunkRepo,
                 events: Publisher = None, infra_status: dict = None):
        self.chat = chat
        self.pref = pref
        self.snap = snap
        self.ltm = ltm
        self.rag_chunk = rag_chunk
        self.events = events or LogPublisher()
        self.infra = infra_status or {}

    def infra_snapshot(self) -> dict:
        return dict(self.infra)
