"""用户偏好仓储"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class Repo:
    """用户偏好仓储接口"""

    def save(self, user_id: str, key: str, value: str):
        raise NotImplementedError

    def load(self, user_id: str) -> Dict[str, str]:
        raise NotImplementedError


class PGRepo(Repo):
    """Postgres 实现"""

    def __init__(self, conn):
        self._conn = conn

    def save(self, user_id: str, key: str, value: str):
        if not self._conn:
            return
        try:
            cur = self._conn.cursor()
            cur.execute(
                """INSERT INTO user_preferences (user_id, key, value) VALUES (%s, %s, %s)
                   ON CONFLICT (user_id, key) DO UPDATE SET value = %s, updated_at = NOW()""",
                (user_id, key, value, value)
            )
            cur.close()
        except Exception as e:
            logger.warning(f"偏好保存失败: {e}")

    def load(self, user_id: str) -> Dict[str, str]:
        if not self._conn:
            return {}
        try:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT key, value FROM user_preferences WHERE user_id = %s",
                (user_id,)
            )
            result = {row[0]: row[1] for row in cur.fetchall()}
            cur.close()
            return result
        except Exception as e:
            logger.warning(f"加载偏好失败: {e}")
            return {}


def new_pg_repo(conn) -> PGRepo:
    return PGRepo(conn)
