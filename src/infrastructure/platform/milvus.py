"""Milvus 向量数据库连接薄封装"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from pymilvus import connections, Collection, utility
    HAS_PYMILVUS = True
except ImportError:
    HAS_PYMILVUS = False


class MilvusClient:
    """Milvus 客户端封装"""

    def __init__(self, host: str = "localhost", port: int = 19530):
        self._host = host
        self._port = port
        self._alias = f"agi_{host}_{port}"
        self._connected = False

    def connect(self) -> bool:
        if not HAS_PYMILVUS:
            return False
        try:
            connections.connect(alias=self._alias, host=self._host, port=str(self._port))
            self._connected = True
            return True
        except Exception as e:
            logger.warning(f"Milvus 连接失败: {e}")
            return False

    @property
    def connected(self) -> bool:
        return self._connected

    def close(self):
        if self._connected:
            try:
                connections.disconnect(self._alias)
            except Exception:
                pass

    def has_collection(self, name: str) -> bool:
        if not self._connected:
            return False
        try:
            return utility.has_collection(name, using=self._alias)
        except Exception:
            return False

    def create_collection(self, name: str, dim: int):
        if not self._connected:
            return
        try:
            from pymilvus import CollectionSchema, FieldSchema, DataType
            fields = [
                FieldSchema(name="pg_id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            ]
            schema = CollectionSchema(fields, description="RAG chunks")
            Collection(name, schema, using=self._alias)
        except Exception as e:
            logger.warning(f"Milvus create collection failed: {e}")

    def insert(self, name: str, pg_ids: list, contents: list, embeddings: list):
        if not self._connected:
            return
        try:
            from pymilvus import Collection
            coll = Collection(name, using=self._alias)
            coll.insert([pg_ids, contents, embeddings])
            coll.flush()
        except Exception as e:
            logger.warning(f"Milvus insert failed: {e}")

    def search(self, name: str, vector: list, top_k: int = 5) -> list:
        if not self._connected:
            return []
        try:
            from pymilvus import Collection
            coll = Collection(name, using=self._alias)
            coll.load()
            results = coll.search(
                data=[vector],
                anns_field="embedding",
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=top_k,
                output_fields=["pg_id"],
            )
            hits = []
            for hits_list in results:
                for hit in hits_list:
                    hits.append({"id": hit.entity.get("pg_id", 0), "distance": hit.distance})
            return hits
        except Exception as e:
            logger.warning(f"Milvus search failed: {e}")
            return []


def connect(cfg) -> tuple:
    """连接 Milvus，返回 (client, status_string)"""
    if not HAS_PYMILVUS:
        logger.warning("pymilvus 未安装")
        return None, "disconnected"

    c = MilvusClient(host=cfg.milvus_host, port=cfg.milvus_port)
    if c.connect():
        logger.info(f"✅ Milvus 已连接: {cfg.milvus_addr()}")
        return c, "connected"
    return None, "disconnected"
