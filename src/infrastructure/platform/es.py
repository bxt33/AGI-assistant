"""Elasticsearch 连接薄封装"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from elasticsearch import Elasticsearch
    HAS_ES = True
except ImportError:
    HAS_ES = False


class ESClient:
    """Elasticsearch 客户端封装"""

    def __init__(self, addresses: list, username: str = "", password: str = ""):
        if username and password:
            self._client = Elasticsearch(addresses, basic_auth=(username, password))
        else:
            self._client = Elasticsearch(addresses)
        self._connected = False

    def ping(self) -> bool:
        try:
            self._connected = self._client.ping()
            return self._connected
        except Exception:
            return False

    @property
    def connected(self) -> bool:
        return self._connected

    def create_index(self, name: str):
        try:
            if not self._client.indices.exists(index=name):
                self._client.indices.create(index=name, body={
                    "mappings": {
                        "properties": {
                            "pg_id": {"type": "long"},
                            "content": {"type": "text", "analyzer": "standard"},
                            "doc_hash": {"type": "keyword"},
                            "chunk_idx": {"type": "integer"},
                        }
                    }
                })
                logger.info(f"✅ ES {name} 索引已创建")
        except Exception as e:
            logger.warning(f"ES create index failed: {e}")

    def index(self, index_name: str, doc_id: str, doc: dict):
        try:
            self._client.index(index=index_name, id=doc_id, document=doc, refresh=False)
        except Exception as e:
            logger.warning(f"ES index failed: {e}")

    def search(self, index_name: str, query: str, top_k: int = 5) -> list:
        try:
            resp = self._client.search(index=index_name, body={
                "size": top_k,
                "query": {"match": {"content": {"query": query}}},
                "_source": ["pg_id"],
            })
            hits = []
            for h in resp["hits"]["hits"]:
                hits.append({
                    "pg_id": h["_source"].get("pg_id", 0),
                    "score": h["_score"],
                })
            return hits
        except Exception as e:
            logger.warning(f"ES search failed: {e}")
            return []

    def delete(self, index_name: str, doc_id: str):
        try:
            self._client.delete(index=index_name, id=doc_id)
        except Exception:
            pass


def connect(cfg) -> tuple:
    """连接 ES，返回 (client, status_string)"""
    if not HAS_ES or not cfg.es_addresses:
        return None, "disconnected"

    try:
        c = ESClient(cfg.es_addresses, cfg.es_username, cfg.es_password)
        if c.ping():
            logger.info(f"✅ Elasticsearch 已连接: {cfg.es_addresses}")
            return c, "connected"
        return None, "disconnected"
    except Exception as e:
        logger.warning(f"Elasticsearch 连接失败: {e}")
        return None, "disconnected"
