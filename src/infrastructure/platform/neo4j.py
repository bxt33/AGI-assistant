"""Neo4j 驱动连接薄封装"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False


class Client:
    """Neo4j 客户端封装"""

    def __init__(self, uri: str = "", user: str = "", password: str = ""):
        self._uri = uri
        self._user = user
        self._password = password
        self._driver = None
        self.available = False

    def connect(self) -> bool:
        if not HAS_NEO4J or not self._uri:
            logger.info(f"Neo4j 未启用（URI={self._uri}）")
            return False

        try:
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password) if self._user else None,
            )
            self._driver.verify_connectivity()
            self.available = True
            self._ensure_constraints()
            logger.info(f"✅ Neo4j 已连接: {self._uri}")
            return True
        except Exception as e:
            logger.warning(f"Neo4j 连接失败: {e}")
            self.available = False
            return False

    def close(self):
        if self._driver:
            self._driver.close()

    def Session(self):
        """返回写入 session（兼容旧 API）"""
        if not self._driver:
            raise RuntimeError("Neo4j not connected")
        return self._driver.session()

    def session(self):
        """返回写入 session（Python 风格）"""
        return self.Session()

    def _ensure_constraints(self):
        """确保 Neo4j 中存在唯一约束/索引（幂等）"""
        if not self.available or not self._driver:
            return
        try:
            with self._driver.session() as sess:
                queries = [
                    "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
                    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
                    "CREATE INDEX memory_node_id IF NOT EXISTS FOR (m:Memory) ON (m.mem_id)",
                ]
                for q in queries:
                    try:
                        sess.run(q)
                    except Exception:
                        pass  # 约束已存在
        except Exception:
            pass


def connect(cfg) -> Client:
    """连接 Neo4j"""
    if not cfg.kg_enabled or not cfg.neo4j_uri:
        return Client()

    c = Client(cfg.neo4j_uri, cfg.neo4j_user, cfg.neo4j_password)
    c.connect()
    return c
