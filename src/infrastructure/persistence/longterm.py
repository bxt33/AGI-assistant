"""长期记忆仓储"""

import json
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class Row:
    def __init__(self, id: int = 0, content: str = "", importance: float = 0.5,
                 embedding: Optional[List[float]] = None, created_at=None,
                 last_accessed=None, category: str = "general",
                 tags: Optional[List[str]] = None, slot_hint: str = ""):
        self.id = id
        self.content = content
        self.importance = importance
        self.embedding = embedding or []
        self.created_at = created_at
        self.last_accessed = last_accessed
        self.category = category
        self.tags = tags or []
        self.slot_hint = slot_hint


class Repo:
    """长期记忆仓储接口"""

    def save(self, content: str, importance: float, embedding_json: Optional[bytes]) -> int:
        raise NotImplementedError

    def save_classified(self, content: str, importance: float,
                        embedding_json: Optional[bytes], category: str = "general",
                        tags: Optional[List[str]] = None,
                        slot_hint: str = "") -> int:
        raise NotImplementedError

    def load(self) -> List[Row]:
        raise NotImplementedError

    def update(self, id: int, content: str, importance: float,
               embedding_json: Optional[bytes]):
        raise NotImplementedError

    def delete(self, ids: List[int]):
        raise NotImplementedError


class PGRepo(Repo):
    """Postgres 实现"""

    def __init__(self, conn):
        self._conn = conn

    def save(self, content: str, importance: float, embedding_json: Optional[bytes]) -> int:
        return self.save_classified(content, importance, embedding_json)

    def save_classified(self, content: str, importance: float,
                        embedding_json: Optional[bytes], category: str = "general",
                        tags: Optional[List[str]] = None,
                        slot_hint: str = "") -> int:
        if not self._conn:
            return -1
        if not category:
            category = "general"
        if tags is None:
            tags = []

        try:
            cur = self._conn.cursor()
            emb_bytes = embedding_json if embedding_json else b"[]"
            cur.execute(
                """INSERT INTO long_term_memory (content, importance, embedding, category, tags, slot_hint)
                   VALUES (%s, %s, %s, %s, %s, NULLIF(%s, '')) RETURNING id""",
                (content, importance, emb_bytes.decode() if isinstance(emb_bytes, bytes) else str(emb_bytes),
                 category, tags, slot_hint or None)
            )
            row = cur.fetchone()
            cur.close()
            return row[0] if row else -1
        except Exception as e:
            logger.warning(f"长期记忆保存失败: {e}")
            return -1

    def load(self) -> List[Row]:
        if not self._conn:
            return []
        try:
            cur = self._conn.cursor()
            cur.execute(
                """SELECT id, content, importance, embedding,
                   COALESCE(created_at, NOW()), COALESCE(last_accessed, NOW()),
                   COALESCE(category, 'general'), COALESCE(tags, '{}'::TEXT[]),
                   COALESCE(slot_hint, '')
                   FROM long_term_memory ORDER BY id"""
            )
            items = []
            for row in cur.fetchall():
                emb = []
                if row[3]:
                    try:
                        emb = json.loads(row[3]) if isinstance(row[3], str) else row[3]
                    except Exception:
                        pass
                items.append(Row(
                    id=row[0], content=row[1], importance=row[2],
                    embedding=emb, created_at=row[4], last_accessed=row[5],
                    category=row[6], tags=list(row[7]) if row[7] else [],
                    slot_hint=row[8] or "",
                ))
            cur.close()
            return items
        except Exception as e:
            logger.warning(f"加载长期记忆失败: {e}")
            return []

    def update(self, id: int, content: str, importance: float,
               embedding_json: Optional[bytes]):
        if not self._conn:
            return
        try:
            cur = self._conn.cursor()
            emb_str = embedding_json.decode() if isinstance(embedding_json, bytes) else str(embedding_json) if embedding_json else "[]"
            cur.execute(
                """UPDATE long_term_memory SET content = %s, importance = %s,
                   embedding = %s, last_accessed = NOW() WHERE id = %s""",
                (content, importance, emb_str, id)
            )
            cur.close()
        except Exception as e:
            logger.warning(f"长期记忆更新失败 (id={id}): {e}")

    def delete(self, ids: List[int]):
        if not self._conn or not ids:
            return
        try:
            cur = self._conn.cursor()
            cur.execute(
                f"DELETE FROM long_term_memory WHERE id IN ({','.join(['%s'] * len(ids))})",
                tuple(ids)
            )
            cur.close()
        except Exception as e:
            logger.warning(f"长期记忆批量删除失败: {e}")


def new_pg_repo(conn) -> PGRepo:
    return PGRepo(conn)
