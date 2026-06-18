"""任务快照仓储"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Repo:
    """任务快照仓储接口"""

    def save(self, task_id: str, state_json: bytes):
        raise NotImplementedError


class PGRepo(Repo):
    """Postgres 实现"""

    def __init__(self, conn):
        self._conn = conn

    def save(self, task_id: str, state_json: bytes):
        if not self._conn:
            return
        try:
            cur = self._conn.cursor()
            state_str = state_json.decode() if isinstance(state_json, bytes) else str(state_json)
            cur.execute(
                """INSERT INTO task_snapshots (task_id, state) VALUES (%s, %s)
                   ON CONFLICT (task_id) DO UPDATE SET state = %s, created_at = NOW()""",
                (task_id, state_str, state_str)
            )
            cur.close()
        except Exception as e:
            logger.warning(f"快照保存失败: {e}")


def new_pg_repo(conn) -> PGRepo:
    return PGRepo(conn)
