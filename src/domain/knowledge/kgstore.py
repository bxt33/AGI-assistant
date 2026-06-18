"""知识图谱存储：Neo4j 图操作封装"""

import logging
from typing import List, Optional, Callable, Dict, Any

from src.domain.knowledge.types import (
    Entity, Relation, GraphSearchResult, ChunkRef, ExtractResult
)
from src.domain.knowledge.extractor import Extractor
from src.infrastructure.platform.neo4j import Client as Neo4jClient

logger = logging.getLogger(__name__)


class KGStore:
    """Neo4j 之上的知识图谱存储：文档索引 + 图检索 + 记忆图扩展"""

    def __init__(self, neo4j_client: Neo4jClient, max_hops: int = 2,
                 kg_weight: float = 0.3,
                 extractor: Optional[Extractor] = None):
        self._neo4j = neo4j_client
        self._max_hops = max(1, min(max_hops, 3))
        self._kg_weight = kg_weight
        self._extractor = extractor

    def available(self) -> bool:
        return self._neo4j is not None and self._neo4j.available

    def client(self) -> Optional[Neo4jClient]:
        return self._neo4j

    def close(self):
        if self._neo4j:
            self._neo4j.close()

    def index_document(self, doc_hash: str, chunks: List[ChunkRef]):
        """为文档 chunks 抽取实体关系并写入图（异步调用）"""
        if not self.available() or not self._extractor:
            return

        for c in chunks:
            result = self._extractor.extract(c.content)
            if not result.entities:
                continue

            # 写入实体节点
            for ent in result.entities:
                ent.doc_hash = doc_hash
                ent.chunk_id = c.id
                ent.pg_id = c.pg_id
                self._upsert_entity(ent)

            # 写入关系边
            for rel in result.relations:
                rel.doc_hash = doc_hash
                rel.chunk_id = c.id
                rel.pg_id = c.pg_id
                self._upsert_relation(rel)

        logger.info(f"知识图谱索引完成：docHash={doc_hash}，chunks={len(chunks)}")

    def delete_document(self, doc_hash: str):
        """删除文档及其关联的孤立节点"""
        if not self.available():
            return
        try:
            sess = self._neo4j.Session()
            try:
                sess.run(
                    "MATCH ()-[r {doc_hash: $doc_hash}]-() DELETE r",
                    {"doc_hash": doc_hash}
                )
                sess.run(
                    "MATCH (e:Entity) WHERE NOT (e)--() AND e.doc_hash = $doc_hash DELETE e",
                    {"doc_hash": doc_hash}
                )
            finally:
                sess.close()
        except Exception as e:
            logger.warning(f"Neo4j 删除文档失败: {e}")

    def search(self, query_text: str, top_k: int = 5) -> List[GraphSearchResult]:
        """图检索：抽取实体 → 子图遍历 → 返回关联 chunk"""
        if not self.available() or not self._extractor:
            return []

        extracted = self._extractor.extract(query_text)
        if not extracted.entities:
            return []

        names = [e.name for e in extracted.entities]
        return self._search_direct(names, top_k)

    def _search_direct(self, names: List[str], top_k: int) -> List[GraphSearchResult]:
        """直接匹配实体所在 chunk（降级方案）"""
        try:
            sess = self._neo4j.Session()
            try:
                records = sess.run(
                    """MATCH (e:Entity) WHERE e.name IN $names AND e.chunk_id IS NOT NULL
                       RETURN e.chunk_id AS cid, COALESCE(e.pg_id, 0) AS pg_id, e.name AS name
                       ORDER BY cid LIMIT $limit""",
                    {"names": names, "limit": top_k}
                )

                seen = set()
                results = []
                for rec in records:
                    pg_id = int(rec.get("pg_id", 0))
                    if pg_id == 0 or pg_id in seen:
                        continue
                    seen.add(pg_id)
                    results.append(GraphSearchResult(
                        chunk_id=int(rec.get("cid", 0)),
                        pg_id=pg_id,
                        score=self._kg_weight,
                        entities=[rec.get("name", "")],
                    ))
                return results
            finally:
                sess.close()
        except Exception as e:
            logger.warning(f"Neo4j 图检索失败: {e}")
            return []

    def _upsert_entity(self, ent: Entity):
        try:
            sess = self._neo4j.Session()
            try:
                sess.run(
                    """MERGE (e:Entity {name: $name})
                       SET e.type = $type, e.doc_hash = $doc_hash,
                           e.chunk_id = $chunk_id, e.pg_id = $pg_id""",
                    {"name": ent.name, "type": ent.type.value,
                     "doc_hash": ent.doc_hash, "chunk_id": ent.chunk_id,
                     "pg_id": ent.pg_id}
                )
            finally:
                sess.close()
        except Exception as e:
            logger.warning(f"Neo4j upsertEntity 失败 ({ent.name}): {e}")

    def _upsert_relation(self, rel: Relation):
        try:
            sess = self._neo4j.Session()
            try:
                sess.run(
                    f"""MERGE (a:Entity {{name: $from}})
                        MERGE (b:Entity {{name: $to}})
                        MERGE (a)-[r:{rel.rel_type} {{doc_hash: $doc_hash}}]->(b)
                        SET r.chunk_id = $chunk_id, r.pg_id = $pg_id""",
                    {"from": rel.from_name, "to": rel.to_name,
                     "doc_hash": rel.doc_hash, "chunk_id": rel.chunk_id,
                     "pg_id": rel.pg_id}
                )
            finally:
                sess.close()
        except Exception as e:
            logger.warning(f"Neo4j upsertRelation 失败 ({rel.from_name}→{rel.to_name}): {e}")
