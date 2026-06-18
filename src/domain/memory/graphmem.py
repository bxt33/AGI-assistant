"""图增强记忆：基于 Neo4j 的记忆图扩展召回。
在长短期记忆上叠加图结构，支持 FOLLOWS / SIMILAR_TO / CAUSES / BELONGS_TO 等关系。
"""

import threading
from typing import List, Optional, Dict, Set
from src.domain.memory.longterm import LongTerm, Item, RecallFilter


class GraphMemory:
    """图增强记忆：通过 Neo4j 客户端建立记忆节点与关系边"""

    def __init__(self, neo4j_client, ltm: LongTerm):
        self._mu = threading.RLock()
        self._neo4j = neo4j_client
        self._ltm = ltm

    def available(self) -> bool:
        return self._neo4j is not None and self._neo4j.available

    def recall_by_filter(self, query: str, query_embedding: Optional[List[float]],
                         flt: RecallFilter) -> List[Item]:
        """图扩展召回：先做 LTM 语义召回，再通过 Neo4j 扩展关联节点"""
        # 先从 LTM 召回
        base = self._ltm.recall_by_filter(query, query_embedding, flt)

        if not self.available() or not base:
            return base

        # 通过 Neo4j 扩展：找到与召回项有关联的其他记忆节点
        try:
            expanded_ids: Set[int] = set()
            sess = self._neo4j.Session()
            try:
                for item in base:
                    records = sess.run(
                        """MATCH (m:Memory {mem_id: $mem_id})-[r:FOLLOWS|SIMILAR_TO|CAUSES|BELONGS_TO]-(n:Memory)
                           RETURN n.mem_id AS mem_id, type(r) AS rel_type
                           LIMIT 5""",
                        {"mem_id": item.id}
                    )
                    for rec in records:
                        nid = rec.get("mem_id")
                        if nid is not None:
                            expanded_ids.add(int(nid))
            finally:
                sess.close()

            # 从 LTM 中查找扩展的节点
            if expanded_ids:
                with self._ltm._mu:
                    for it in self._ltm._items:
                        if it.id in expanded_ids and it not in base:
                            base.append(it)
        except Exception:
            pass  # Neo4j 不可用时静默降级

        return base

    def index_memory(self, item: Item, prev_item: Optional[Item] = None):
        """在 Neo4j 中为记忆条目建立节点和关系"""
        if not self.available():
            return

        try:
            sess = self._neo4j.Session()
            try:
                # 创建记忆节点
                sess.run(
                    """MERGE (m:Memory {mem_id: $mem_id})
                       SET m.content = $content, m.importance = $importance,
                           m.category = $category""",
                    {"mem_id": item.id, "content": item.content[:200],
                     "importance": item.importance, "category": item.category}
                )
                # 建立 FOLLOWS 关系（时序）
                if prev_item:
                    sess.run(
                        """MATCH (a:Memory {mem_id: $prev_id}), (b:Memory {mem_id: $curr_id})
                           MERGE (a)-[r:FOLLOWS]->(b)""",
                        {"prev_id": prev_item.id, "curr_id": item.id}
                    )
            finally:
                sess.close()
        except Exception:
            pass

    def delete_memory_node(self, mem_id: int):
        """删除记忆节点"""
        if not self.available():
            return
        try:
            sess = self._neo4j.Session()
            try:
                sess.run("MATCH (m:Memory {mem_id: $mem_id}) DETACH DELETE m",
                         {"mem_id": mem_id})
            finally:
                sess.close()
        except Exception:
            pass

    def index_similar_relation(self, item_a: Item, item_b: Item, similarity: float):
        """建立 SIMILAR_TO 关系"""
        if not self.available():
            return
        try:
            sess = self._neo4j.Session()
            try:
                sess.run(
                    """MATCH (a:Memory {mem_id: $id_a}), (b:Memory {mem_id: $id_b})
                       MERGE (a)-[r:SIMILAR_TO]->(b)
                       SET r.score = $score""",
                    {"id_a": item_a.id, "id_b": item_b.id, "score": similarity}
                )
            finally:
                sess.close()
        except Exception:
            pass

    def merge_graph_nodes(self, keep_id: int, remove_ids: List[int]):
        """合并节点：将 remove_ids 的关系转移给 keep_id 后删除"""
        if not self.available():
            return
        try:
            sess = self._neo4j.Session()
            try:
                for rid in remove_ids:
                    sess.run(
                        """MATCH (old:Memory {mem_id: $old_id})-[r]-(n)
                           WHERE n.mem_id <> $keep_id
                           DELETE r""",
                        {"old_id": rid, "keep_id": keep_id}
                    )
                    sess.run(
                        "MATCH (m:Memory {mem_id: $old_id}) DETACH DELETE m",
                        {"old_id": rid}
                    )
            finally:
                sess.close()
        except Exception:
            pass
