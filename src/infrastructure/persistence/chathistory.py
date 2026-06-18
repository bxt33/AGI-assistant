"""聊天记录仓储"""

import logging
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Entry:
    role: str = ""
    content: str = ""
    created_at: str = ""


class Repo:
    """聊天记录仓储接口"""

    def save(self, role: str, content: str):
        raise NotImplementedError

    def load(self, limit: int = 20) -> List[Entry]:
        raise NotImplementedError


class PGRepo(Repo):
    """Postgres 实现"""

    def __init__(self, conn):
        self._conn = conn

    def save(self, role: str, content: str):
        if not self._conn:
            return
        try:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO chat_history (role, content) VALUES (%s, %s)",
                (role, content)
            )
            cur.close()
        except Exception as e:
            logger.warning(f"聊天记录保存失败: {e}")

    def load(self, limit: int = 20) -> List[Entry]:
        if not self._conn:
            return []
        try:
            cur = self._conn.cursor()
            cur.execute(
                """SELECT role, content, TO_CHAR(created_at, 'HH24:MI:SS')
                   FROM chat_history ORDER BY id DESC LIMIT %s""",
                (limit,)
            )
            rows = cur.fetchall()
            cur.close()
            # 反转为时间正序
            entries = [Entry(role=r[0], content=r[1], created_at=r[2]) for r in rows]
            entries.reverse()
            return entries
        except Exception as e:
            logger.warning(f"加载聊天记录失败: {e}")
            return []


def new_pg_repo(conn) -> PGRepo:
    return PGRepo(conn)
