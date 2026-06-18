"""PostgreSQL 连接管理 + Schema Bootstrap"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import psycopg2
    import psycopg2.pool
    HAS_PSYCOPG = True
except ImportError:
    HAS_PSYCOPG = False


def connect(cfg) -> tuple:
    """连接 PG，返回 (connection, status_string)"""
    if not HAS_PSYCOPG:
        logger.warning("psycopg2 未安装，PG 连接降级")
        return None, "disconnected"

    try:
        conn = psycopg2.connect(cfg.pg_dsn())
        conn.autocommit = True
        logger.info(f"✅ PostgreSQL 已连接: {cfg.pg_host}:{cfg.pg_port}")
        return conn, "connected"
    except Exception as e:
        logger.warning(f"PostgreSQL 连接失败: {e}")
        return None, "disconnected"


def bootstrap_schema(conn):
    """幂等地创建/升级所有业务表"""
    if conn is None:
        return

    ddls = [
        """CREATE TABLE IF NOT EXISTS user_preferences (
            user_id    TEXT NOT NULL,
            key        TEXT NOT NULL,
            value      TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (user_id, key)
        )""",
        """CREATE TABLE IF NOT EXISTS task_snapshots (
            task_id    TEXT PRIMARY KEY,
            state      JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )""",
        """CREATE TABLE IF NOT EXISTS chat_history (
            id         SERIAL PRIMARY KEY,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )""",
        """CREATE TABLE IF NOT EXISTS long_term_memory (
            id            SERIAL PRIMARY KEY,
            content       TEXT NOT NULL,
            importance    FLOAT NOT NULL DEFAULT 0.5,
            embedding     JSONB,
            created_at    TIMESTAMP DEFAULT NOW(),
            last_accessed TIMESTAMP DEFAULT NOW(),
            category      TEXT NOT NULL DEFAULT 'general',
            tags          TEXT[] NOT NULL DEFAULT '{}',
            slot_hint     TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS rag_chunks (
            id          BIGSERIAL PRIMARY KEY,
            doc_hash    TEXT NOT NULL,
            chunk_idx   INT NOT NULL,
            content     TEXT NOT NULL,
            parent_content TEXT,
            embedding   JSONB,
            created_at  TIMESTAMP DEFAULT NOW(),
            UNIQUE(doc_hash, chunk_idx)
        )""",
        # 索引
        """CREATE INDEX IF NOT EXISTS idx_ltm_category ON long_term_memory(category)""",
        """CREATE INDEX IF NOT EXISTS idx_ltm_tags ON long_term_memory USING GIN(tags)""",
    ]

    cur = conn.cursor()
    try:
        for ddl in ddls:
            try:
                cur.execute(ddl)
            except Exception as e:
                logger.warning(f"PG DDL 执行跳过: {e}")
    finally:
        cur.close()

    logger.info("✅ PostgreSQL 表结构已初始化")
