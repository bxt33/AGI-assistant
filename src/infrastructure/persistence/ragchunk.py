"""RAG chunk 统一仓储：扇出到 PG + Milvus + ES"""

import json
import logging
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Row:
    id: int = 0
    content: str = ""
    parent_content: str = ""


class Repo:
    """RAG chunk 综合仓储接口"""

    def SavePG(self, doc_hash: str, chunk_idx: int, content: str,
               embedding_json: Optional[str]) -> tuple:
        raise NotImplementedError

    def LoadAll(self) -> List[Row]:
        raise NotImplementedError

    def LoadByIDs(self, ids: List[int]) -> List[Row]:
        raise NotImplementedError

    def SearchES(self, query: str, top_k: int) -> List:
        raise NotImplementedError

    def SearchMilvus(self, vector: List[float], top_k: int) -> List:
        raise NotImplementedError

    def IndexES(self, pg_id: int, content: str, doc_hash: str, chunk_idx: int):
        raise NotImplementedError

    def InsertMilvus(self, pg_ids: List[int], contents: List[str],
                     embeddings: List[List[float]]):
        raise NotImplementedError

    def Delete(self, doc_hash: str):
        raise NotImplementedError

    def Init(self, dim: int):
        raise NotImplementedError

    def MilvusAvailable(self) -> bool:
        raise NotImplementedError

    def ESAvailable(self) -> bool:
        raise NotImplementedError


class Store(Repo):
    """多后端组合实现"""

    def __init__(self, pg_conn=None, milvus_client=None, es_client=None):
        self._pg = pg_conn
        self._milvus = milvus_client
        self._es = es_client

    def MilvusAvailable(self) -> bool:
        return self._milvus is not None and getattr(self._milvus, 'connected', False)

    def ESAvailable(self) -> bool:
        return self._es is not None and getattr(self._es, 'connected', False)

    def SavePG(self, doc_hash: str, chunk_idx: int, content: str,
               embedding_json: Optional[str]) -> tuple:
        if not self._pg:
            return -1, "postgres not connected"
        try:
            cur = self._pg.cursor()
            parent_content = ""
            cur.execute(
                """INSERT INTO rag_chunks (doc_hash, chunk_idx, content, parent_content, embedding)
                   VALUES (%s, %s, %s, NULLIF(%s, ''), %s)
                   ON CONFLICT (doc_hash, chunk_idx) DO UPDATE
                     SET content = EXCLUDED.content,
                         parent_content = EXCLUDED.parent_content,
                         embedding = EXCLUDED.embedding
                   RETURNING id""",
                (doc_hash, chunk_idx, content, parent_content,
                 embedding_json if embedding_json else None)
            )
            row = cur.fetchone()
            cur.close()
            return (row[0], None) if row else (-1, "no id returned")
        except Exception as e:
            return -1, str(e)

    def LoadAll(self) -> List[Row]:
        if not self._pg:
            return []
        try:
            cur = self._pg.cursor()
            cur.execute(
                "SELECT id, content, COALESCE(parent_content, '') FROM rag_chunks ORDER BY id"
            )
            rows = [Row(id=r[0], content=r[1], parent_content=r[2]) for r in cur.fetchall()]
            cur.close()
            return rows
        except Exception as e:
            logger.warning(f"LoadAll rag chunks failed: {e}")
            return []

    def LoadByIDs(self, ids: List[int]) -> List[Row]:
        if not self._pg or not ids:
            return []
        try:
            cur = self._pg.cursor()
            placeholders = ",".join(["%s"] * len(ids))
            cur.execute(
                f"SELECT id, content, COALESCE(parent_content, '') FROM rag_chunks WHERE id IN ({placeholders})",
                tuple(ids)
            )
            rows = [Row(id=r[0], content=r[1], parent_content=r[2]) for r in cur.fetchall()]
            cur.close()
            return rows
        except Exception as e:
            logger.warning(f"LoadByIDs failed: {e}")
            return []

    def SearchES(self, query: str, top_k: int) -> list:
        if not self.ESAvailable():
            return []
        try:
            return self._es.search("rag_chunks", query, top_k)
        except Exception as e:
            logger.warning(f"ES search failed: {e}")
            return []

    def SearchMilvus(self, vector: List[float], top_k: int) -> list:
        if not self.MilvusAvailable():
            return []
        try:
            return self._milvus.search("rag_chunks", vector, top_k)
        except Exception as e:
            logger.warning(f"Milvus search failed: {e}")
            return []

    def IndexES(self, pg_id: int, content: str, doc_hash: str, chunk_idx: int):
        if not self.ESAvailable():
            return
        try:
            doc = {"pg_id": pg_id, "content": content, "doc_hash": doc_hash, "chunk_idx": chunk_idx}
            self._es.index("rag_chunks", str(pg_id), doc)
        except Exception as e:
            logger.warning(f"ES index failed: {e}")

    def InsertMilvus(self, pg_ids: List[int], contents: List[str],
                     embeddings: List[List[float]]):
        if not self.MilvusAvailable():
            return
        try:
            self._milvus.insert("rag_chunks", pg_ids, contents, embeddings)
        except Exception as e:
            logger.warning(f"Milvus insert failed: {e}")

    def Delete(self, doc_hash: str):
        # PG 删除
        pg_ids = []
        if self._pg:
            try:
                cur = self._pg.cursor()
                cur.execute("SELECT id FROM rag_chunks WHERE doc_hash = %s", (doc_hash,))
                pg_ids = [r[0] for r in cur.fetchall()]
                if pg_ids:
                    placeholders = ",".join(["%s"] * len(pg_ids))
                    cur.execute(
                        f"DELETE FROM rag_chunks WHERE id IN ({placeholders})",
                        tuple(pg_ids)
                    )
                cur.close()
            except Exception as e:
                logger.warning(f"PG delete failed: {e}")

        # ES 删除
        if self.ESAvailable() and pg_ids:
            for pid in pg_ids:
                try:
                    self._es.delete("rag_chunks", str(pid))
                except Exception:
                    pass

        # Milvus 删除
        if self.MilvusAvailable() and pg_ids:
            try:
                expr = f"pg_id in [{','.join(str(i) for i in pg_ids)}]"
                self._milvus.delete("rag_chunks", expr)
            except Exception:
                pass

    def Init(self, dim: int):
        if self._es:
            try:
                self._es.create_index("rag_chunks")
            except Exception as e:
                logger.warning(f"ES init failed: {e}")
        if self._milvus:
            try:
                if not self._milvus.has_collection("rag_chunks"):
                    self._milvus.create_collection("rag_chunks", dim)
            except Exception as e:
                logger.warning(f"Milvus init failed: {e}")


def new_store(pg_conn=None, milvus_client=None, es_client=None) -> Store:
    return Store(pg_conn, milvus_client, es_client)
