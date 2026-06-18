#!/usr/bin/env python3
"""AGI Saber — Python 版本主入口

=============================================================================
后端启动流程（简化版）：
  1. 加载 config.yaml 配置（LLM密钥、数据库地址、RAG参数等）
  2. 连接基础设施（PostgreSQL / Milvus / ES / Kafka）— 每个独立，失败不阻塞
  3. 组装依赖注入字典 → 传入 UnifiedAgent（整个系统的核心大脑）
  4. 创建 FastAPI app，把 agent 挂到 HTTP 路由上
  5. 启动 uvicorn，监听 8090 端口

关键对象关系：
  DefaultConfig()  ──►  UnifiedAgent(cfg, deps)  ──►  create_app(agent)  ──►  uvicorn.run(app)
       ↑                        ↑                           ↑
  读 config.yaml          核心调度大脑              挂载 10 个 API 端点
                          (core_agent.py)            (handler.py)
=============================================================================
"""

import logging
import sys

import uvicorn

from config.config import DefaultConfig
from src.application.chat.core_agent import UnifiedAgent
from src.infrastructure.platform.postgres import connect as pg_connect, bootstrap_schema
from src.infrastructure.platform.milvus import connect as milvus_connect
from src.infrastructure.platform.es import connect as es_connect
from src.infrastructure.platform.kafka import connect as kafka_connect
from src.infrastructure.persistence.chathistory import new_pg_repo as chat_repo
from src.infrastructure.persistence.longterm import new_pg_repo as ltm_repo
from src.infrastructure.persistence.preference import new_pg_repo as pref_repo
from src.infrastructure.persistence.snapshot import new_pg_repo as snap_repo
from src.infrastructure.persistence.ragchunk import new_store as rag_store
from src.infrastructure.eventbus import KafkaPublisher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def print_banner(cfg, infra_status: dict):
    """打印启动横幅，展示服务地址、模型信息、基础设施连接状态"""
    addr = f":{cfg.ServerPort}"
    print("=" * 50)
    print("AGI Saber · 智能助手启动成功")
    print("=" * 50)
    print(f"[INFO] Service       http://localhost{addr}")
    print(f"[INFO] 通用模型      {cfg.LLMModel}")
    print(f"[INFO] Embedding     {cfg.EmbeddingModel}")
    print("-" * 50)
    for name, status in infra_status.items():
        print(f"[INFO] {name:14s} {status}")
    print("-" * 50)
    print("[READY] 道阻且长，行则将至。")
    print("=" * 50)


def main():
    """主启动函数

    整个后端只有一个入口：这里初始化所有依赖，造出 UnifiedAgent，启动 HTTP 服务。
    之后的一切（路由、推理、记忆、RAG）都由 UnifiedAgent 内部调度。
    """

    # ──────────────────────────────────────────────────────────────
    # 步骤 1：加载配置（config/config.yaml → APIConfig 数据类）
    #   - LLM 模型名和 API Key
    #   - Embedding 模型配置
    #   - PG / Milvus / ES / Kafka / Neo4j 连接信息
    #   - RAG 参数（分块大小、检索 TopK、RRF 权重等）
    #   - 记忆参数（短期窗口大小、长期衰减率、合并阈值等）
    #   - 沙箱参数（Docker/Local/Mock 后端选择、资源限制）
    # ──────────────────────────────────────────────────────────────
    cfg = DefaultConfig()
    infra_status = {}

    # ──────────────────────────────────────────────────────────────
    # 步骤 2：连接基础设施（每路独立 try-catch，失败自动降级，不阻塞启动）
    #
    #   关键设计理念：所有基础设施都是"可选"的。
    #   - 没有 Docker → Milvus/ES 连不上 → RAG 降级为空（功能不受影响）
    #   - 没有 PG → 记忆不持久化，但内存模式照常工作
    #   - 没有 Kafka → 事件用 LogPublisher 替代（只打日志）
    #
    #   每个 connect() 返回 (client_or_none, status_string)
    #   成功 → (client对象,  "connected")
    #   失败 → (None,        "degraded: reason")
    # ──────────────────────────────────────────────────────────────
    logger.info("🔧 正在连接基础设施...")

    # Milvus：向量数据库，存文档的 Embedding 向量，支持语义相似度搜索
    milvus_client, milvus_status = milvus_connect(cfg.MilvusConfig)
    infra_status["milvus"] = milvus_status

    # PostgreSQL：关系型数据库，持久化聊天记录 / 长期记忆 / 用户偏好 / 任务快照
    pg_conn, pg_status = pg_connect(cfg.PostgresConfig)
    if pg_conn:
        bootstrap_schema(pg_conn)  # 自动建表（IF NOT EXISTS，幂等）
    infra_status["postgresql"] = pg_status

    # Elasticsearch：全文搜索引擎，提供 BM25 关键词检索
    es_client, es_status = es_connect(cfg.ESConfig)
    infra_status["elasticsearch"] = es_status

    # Kafka：消息队列，发布 Agent 事件（对话开始/结束、工具调用等）
    kafka_producer, kafka_status = kafka_connect(cfg.KafkaConfig)
    infra_status["kafka"] = kafka_status

    # ──────────────────────────────────────────────────────────────
    # 步骤 3：组装依赖注入字典（deps）
    #
    #   这里体现了 Clean Architecture 的"依赖注入"思想：
    #   UnifiedAgent 不自己创建数据库连接，而是通过 deps 字典"注入"。
    #   好处：测试时可以换成 Mock 实现，不同环境可以连不同的数据库。
    #
    #   deps 包含 5 个 Repository + 1 个 EventPublisher + 基础设施状态：
    #     chat      → 聊天记录的增删查
    #     pref      → 用户偏好的存取（"用户叫张三""喜欢简洁回答"）
    #     snap      → ReAct 任务快照（中断恢复用）
    #     ltm       → 长期记忆持久化
    #     rag_chunk  → RAG 文档块的多后端存储（PG+Milvus+ES 三写）
    #     events    → 事件发布（Kafka 或 Log 降级）
    # ──────────────────────────────────────────────────────────────
    deps = {
        "chat": chat_repo(pg_conn),           # 聊天历史 → PostgreSQL
        "pref": pref_repo(pg_conn),           # 用户偏好 → PostgreSQL
        "snap": snap_repo(pg_conn),           # 任务快照 → PostgreSQL
        "ltm": ltm_repo(pg_conn),             # 长期记忆 → PostgreSQL
        "rag_chunk": rag_store(pg_conn, milvus_client, es_client),  # RAG 三后端
        "events": KafkaPublisher(kafka_producer, kafka_status == "connected"),
        "infra_status": infra_status,         # 基础设施连接状态（给前端展示用）
    }

    # ──────────────────────────────────────────────────────────────
    # 步骤 4：创建 UnifiedAgent — 整个系统的核心大脑
    #
    #   UnifiedAgent 构造时会自动完成：
    #   - 初始化 LLM 客户端（DeepSeek / OpenAI 兼容接口）
    #   - 初始化 RAG 引擎（三路混合检索：Milvus + ES + Neo4j）
    #   - 初始化三层记忆系统（短期/长期/偏好 + Neo4j 图增强）
    #   - 初始化工具注册表（时间/天气/搜索/命令执行/MCP）
    #   - 初始化沙箱（Docker/Local/Mock 三选一）
    #   - 初始化 Schema 驱动提示词组装器（promptctx/）
    #   - 从 PG 恢复历史记忆和偏好（跨会话持久化）
    #
    #   之后所有的对话请求都由 agent.Process(message, opts) 处理
    #   详见：src/application/chat/core_agent.py
    # ──────────────────────────────────────────────────────────────
    agent = UnifiedAgent(cfg, deps)

    # ──────────────────────────────────────────────────────────────
    # 步骤 5：创建 FastAPI 应用，挂载路由
    #
    #   create_app() 做的事（见 src/interfaces/http/handler.py）：
    #   - 注册 10 个 HTTP 端点（/api/chat, /api/chat/stream, /api/upload 等）
    #   - 注册 CORS 中间件（允许前端跨域访问）
    #   - 挂载静态文件服务（frontend/index.html）
    #
    #   关键：agent 对象被传入 create_app，所有 HTTP 请求最终都调 agent.Process()
    # ──────────────────────────────────────────────────────────────
    from src.interfaces.http.handler import create_app
    app = create_app(agent, cfg)

    # ──────────────────────────────────────────────────────────────
    # 步骤 6：启动 uvicorn HTTP 服务器
    #   - 监听 0.0.0.0:8090（config.yaml 中可改端口）
    #   - 默认 log_level=info
    # ──────────────────────────────────────────────────────────────
    print_banner(cfg, infra_status)

    addr = f"0.0.0.0:{cfg.ServerPort}"
    uvicorn.run(app, host="0.0.0.0", port=int(cfg.ServerPort), log_level="info")


if __name__ == "__main__":
    main()
