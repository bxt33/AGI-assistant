"""配置加载：从 config.yaml 读取并填充默认值"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ServerConfig:
    server_port: str = "8090"


@dataclass
class LLMConfig:
    llm_api_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    temperature: float = 0.7

    def is_real_llm(self) -> bool:
        return self.llm_api_key != ""


@dataclass
class EmbeddingConfig:
    embedding_api_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = ""

    def is_real_embedding(self) -> bool:
        return self.embedding_api_key != ""


@dataclass
class MilvusConfig:
    milvus_host: str = "localhost"
    milvus_port: int = 19530

    def milvus_addr(self) -> str:
        return f"{self.milvus_host}:{self.milvus_port}"


@dataclass
class PostgresConfig:
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_user: str = "postgres"
    pg_password: str = "postgres"
    pg_database: str = "agi_assistant"

    def pg_dsn(self) -> str:
        return f"postgresql://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_database}"


@dataclass
class ESConfig:
    es_addresses: List[str] = field(default_factory=list)
    es_username: str = ""
    es_password: str = ""


@dataclass
class KafkaConfig:
    kafka_brokers: List[str] = field(default_factory=list)
    kafka_topic: str = "agi-events"


@dataclass
class Neo4jConfig:
    neo4j_uri: str = ""
    neo4j_user: str = ""
    neo4j_password: str = ""
    kg_max_hops: int = 2
    kg_weight: float = 0.3
    kg_enabled: bool = False


@dataclass
class StorageConfig:
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    es: ESConfig = field(default_factory=ESConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)


@dataclass
class RAGConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    rrf_constant_k: int = 60
    semantic_weight: float = 0.7
    enable_hybrid_search: bool = True
    rag_milvus_dim: int = 1024
    rag_rewrite_enabled: bool = True
    rag_rewrite_num_queries: int = 3
    rag_rerank_enabled: bool = True
    rag_rerank_preview_len: int = 200


@dataclass
class MemoryConfig:
    short_term_max_turns: int = 10
    long_term_top_k: int = 5
    memory_consolidation_similarity: float = 0.80
    memory_consolidation_dedup: float = 0.95
    memory_consolidation_ttl_days: int = 30
    memory_consolidation_decay_rate: float = 0.995
    memory_consolidation_min_import: float = 0.3
    memory_consolidation_trigger: int = 5


@dataclass
class HarnessConfig:
    max_retries: int = 3
    retry_delay_ms: int = 200
    step_timeout_ms: int = 30000
    max_iterations: int = 10


@dataclass
class SearchConfig:
    search_api_key: str = ""
    search_api_url: str = ""


@dataclass
class SandboxConfig:
    sandbox_enabled: bool = True
    sandbox_backend: str = "docker"
    sandbox_image: str = "ubuntu:22.04"
    sandbox_timeout_ms: int = 30000
    sandbox_max_output: int = 65536
    sandbox_memory_mb: int = 256
    sandbox_cpu_percent: int = 50
    sandbox_max_pids: int = 64
    sandbox_net_disabled: bool = True
    sandbox_read_only: bool = True


@dataclass
class SecurityConfig:
    sec_max_cmd_length: int = 500
    sec_allowlist_mode: bool = False
    sec_allowlist: List[str] = field(default_factory=list)


@dataclass
class GraphRuntimeConfig:
    graph_max_parallel: int = 2
    graph_race_timeout_ms: int = 30000
    graph_enable_racing: bool = True


@dataclass
class APIConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    graph_runtime: GraphRuntimeConfig = field(default_factory=GraphRuntimeConfig)

    # Convenience property aliases for backward compatibility
    @property
    def ServerPort(self) -> str: return self.server.server_port
    @property
    def LLMAPIUrl(self) -> str: return self.llm.llm_api_url
    @property
    def LLMAPIKey(self) -> str: return self.llm.llm_api_key
    @property
    def LLMModel(self) -> str: return self.llm.llm_model
    @property
    def Temperature(self) -> float: return self.llm.temperature
    @property
    def EmbeddingAPIUrl(self) -> str: return self.embedding.embedding_api_url
    @property
    def EmbeddingAPIKey(self) -> str: return self.embedding.embedding_api_key
    @property
    def EmbeddingModel(self) -> str: return self.embedding.embedding_model
    @property
    def MilvusHost(self) -> str: return self.storage.milvus.milvus_host
    @property
    def MilvusPort(self) -> int: return self.storage.milvus.milvus_port
    @property
    def PGHost(self) -> str: return self.storage.postgres.pg_host
    @property
    def PGPort(self) -> int: return self.storage.postgres.pg_port
    @property
    def IsRealLLM(self) -> bool: return self.llm.is_real_llm()
    @property
    def Neo4jConfig(self): return self.storage.neo4j
    @property
    def KGMaxHops(self) -> int: return self.storage.neo4j.kg_max_hops
    @property
    def KGWeight(self) -> float: return self.storage.neo4j.kg_weight
    @property
    def KGEnabled(self) -> bool: return self.storage.neo4j.kg_enabled
    @property
    def SearchAPIKey(self) -> str: return self.search.search_api_key
    @property
    def SearchAPIURL(self) -> str: return self.search.search_api_url
    @property
    def MilvusConfig(self): return self.storage.milvus
    @property
    def PostgresConfig(self): return self.storage.postgres
    @property
    def ESConfig(self): return self.storage.es
    @property
    def KafkaConfig(self): return self.storage.kafka

    # RAG properties
    @property
    def ChunkSize(self) -> int: return self.rag.chunk_size
    @property
    def ChunkOverlap(self) -> int: return self.rag.chunk_overlap
    @property
    def TopK(self) -> int: return self.rag.top_k
    @property
    def RRFConstantK(self) -> int: return self.rag.rrf_constant_k
    @property
    def SemanticWeight(self) -> float: return self.rag.semantic_weight
    @property
    def EnableHybridSearch(self) -> bool: return self.rag.enable_hybrid_search
    @property
    def RAGMilvusDim(self) -> int: return self.rag.rag_milvus_dim
    @property
    def RAGRewriteEnabled(self) -> bool: return self.rag.rag_rewrite_enabled
    @property
    def RAGRewriteNumQueries(self) -> int: return self.rag.rag_rewrite_num_queries
    @property
    def RAGRerankEnabled(self) -> bool: return self.rag.rag_rerank_enabled
    @property
    def RAGRerankPreviewLen(self) -> int: return self.rag.rag_rerank_preview_len

    # Memory properties
    @property
    def ShortTermMaxTurns(self) -> int: return self.memory.short_term_max_turns
    @property
    def LongTermTopK(self) -> int: return self.memory.long_term_top_k
    @property
    def MemoryConsolidationSimilarity(self) -> float: return self.memory.memory_consolidation_similarity
    @property
    def MemoryConsolidationDedup(self) -> float: return self.memory.memory_consolidation_dedup
    @property
    def MemoryConsolidationTTLDays(self) -> int: return self.memory.memory_consolidation_ttl_days
    @property
    def MemoryConsolidationDecayRate(self) -> float: return self.memory.memory_consolidation_decay_rate
    @property
    def MemoryConsolidationMinImport(self) -> float: return self.memory.memory_consolidation_min_import
    @property
    def MemoryConsolidationTrigger(self) -> int: return self.memory.memory_consolidation_trigger

    # Harness properties
    @property
    def MaxRetries(self) -> int: return self.harness.max_retries
    @property
    def RetryDelayMs(self) -> int: return self.harness.retry_delay_ms
    @property
    def StepTimeoutMs(self) -> int: return self.harness.step_timeout_ms
    @property
    def MaxIterations(self) -> int: return self.harness.max_iterations

    # Sandbox properties
    @property
    def SandboxEnabled(self) -> bool: return self.sandbox.sandbox_enabled
    @property
    def SandboxBackend(self) -> str: return self.sandbox.sandbox_backend
    @property
    def SandboxImage(self) -> str: return self.sandbox.sandbox_image
    @property
    def SandboxTimeoutMs(self) -> int: return self.sandbox.sandbox_timeout_ms
    @property
    def SandboxMaxOutput(self) -> int: return self.sandbox.sandbox_max_output
    @property
    def SandboxMemoryMB(self) -> int: return self.sandbox.sandbox_memory_mb
    @property
    def SandboxCPUPercent(self) -> int: return self.sandbox.sandbox_cpu_percent
    @property
    def SandboxMaxPIDs(self) -> int: return self.sandbox.sandbox_max_pids
    @property
    def SandboxNetDisabled(self) -> bool: return self.sandbox.sandbox_net_disabled
    @property
    def SandboxReadOnly(self) -> bool: return self.sandbox.sandbox_read_only

    # Security properties
    @property
    def SecMaxCmdLength(self) -> int: return self.security.sec_max_cmd_length
    @property
    def SecAllowlistMode(self) -> bool: return self.security.sec_allowlist_mode
    @property
    def SecAllowlist(self) -> List[str]: return self.security.sec_allowlist

    # Graph runtime properties
    @property
    def GraphMaxParallel(self) -> int: return self.graph_runtime.graph_max_parallel
    @property
    def GraphRaceTimeoutMs(self) -> int: return self.graph_runtime.graph_race_timeout_ms
    @property
    def GraphEnableRacing(self) -> bool: return self.graph_runtime.graph_enable_racing


def _apply_defaults(cfg: APIConfig):
    """为零值字段填充合理默认值"""
    # RAG defaults
    if cfg.rag.rrf_constant_k <= 0:
        cfg.rag.rrf_constant_k = 60
    if cfg.rag.semantic_weight <= 0:
        cfg.rag.semantic_weight = 0.7
    if cfg.rag.rag_milvus_dim <= 0:
        cfg.rag.rag_milvus_dim = 1024
    if cfg.rag.rag_rewrite_num_queries <= 0:
        cfg.rag.rag_rewrite_num_queries = 3
    if cfg.rag.rag_rerank_preview_len <= 0:
        cfg.rag.rag_rerank_preview_len = 200

    # Memory defaults
    if cfg.memory.memory_consolidation_similarity <= 0:
        cfg.memory.memory_consolidation_similarity = 0.80
    if cfg.memory.memory_consolidation_dedup <= 0:
        cfg.memory.memory_consolidation_dedup = 0.95
    if cfg.memory.memory_consolidation_ttl_days <= 0:
        cfg.memory.memory_consolidation_ttl_days = 30
    if cfg.memory.memory_consolidation_decay_rate <= 0:
        cfg.memory.memory_consolidation_decay_rate = 0.995
    if cfg.memory.memory_consolidation_min_import <= 0:
        cfg.memory.memory_consolidation_min_import = 0.3
    if cfg.memory.memory_consolidation_trigger <= 0:
        cfg.memory.memory_consolidation_trigger = 5

    # Neo4j defaults
    if cfg.storage.neo4j.kg_max_hops <= 0:
        cfg.storage.neo4j.kg_max_hops = 2
    if cfg.storage.neo4j.kg_weight <= 0:
        cfg.storage.neo4j.kg_weight = 0.3

    # Sandbox defaults
    if not cfg.sandbox.sandbox_backend:
        cfg.sandbox.sandbox_backend = "docker"
    if not cfg.sandbox.sandbox_image:
        cfg.sandbox.sandbox_image = "ubuntu:22.04"
    if cfg.sandbox.sandbox_timeout_ms <= 0:
        cfg.sandbox.sandbox_timeout_ms = 30000
    if cfg.sandbox.sandbox_max_output <= 0:
        cfg.sandbox.sandbox_max_output = 65536
    if cfg.sandbox.sandbox_memory_mb <= 0:
        cfg.sandbox.sandbox_memory_mb = 256
    if cfg.sandbox.sandbox_cpu_percent <= 0:
        cfg.sandbox.sandbox_cpu_percent = 50
    if cfg.sandbox.sandbox_max_pids <= 0:
        cfg.sandbox.sandbox_max_pids = 64

    # Security defaults
    if cfg.security.sec_max_cmd_length <= 0:
        cfg.security.sec_max_cmd_length = 500

    # Graph runtime defaults
    if cfg.graph_runtime.graph_max_parallel <= 0:
        cfg.graph_runtime.graph_max_parallel = 2
    if cfg.graph_runtime.graph_race_timeout_ms <= 0:
        cfg.graph_runtime.graph_race_timeout_ms = 30000


def DefaultConfig(config_path: str = "config/config.yaml") -> APIConfig:
    """从 config/config.yaml 加载配置，填充默认值"""
    if not os.path.exists(config_path):
        print(f"⚠️  未找到 {config_path}，使用默认配置")
        cfg = APIConfig()
        _apply_defaults(cfg)
        return cfg

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    cfg = APIConfig()

    # Server
    if "server" in data:
        cfg.server.server_port = str(data["server"].get("port", "8090"))

    # LLM
    if "llm" in data:
        llm = data["llm"]
        cfg.llm.llm_api_url = llm.get("api_url", "")
        cfg.llm.llm_api_key = llm.get("api_key", "")
        cfg.llm.llm_model = llm.get("model", "")
        cfg.llm.temperature = llm.get("temperature", 0.7)

    # Embedding
    if "embedding" in data:
        emb = data["embedding"]
        cfg.embedding.embedding_api_url = emb.get("api_url", "")
        cfg.embedding.embedding_api_key = emb.get("api_key", "")
        cfg.embedding.embedding_model = emb.get("model", "")

    # Milvus
    if "milvus" in data:
        mv = data["milvus"]
        cfg.storage.milvus.milvus_host = mv.get("host", "localhost")
        cfg.storage.milvus.milvus_port = mv.get("port", 19530)

    # PostgreSQL
    if "postgres" in data:
        pg = data["postgres"]
        cfg.storage.postgres.pg_host = pg.get("host", "localhost")
        cfg.storage.postgres.pg_port = pg.get("port", 5432)
        cfg.storage.postgres.pg_user = pg.get("user", "postgres")
        cfg.storage.postgres.pg_password = pg.get("password", "postgres")
        cfg.storage.postgres.pg_database = pg.get("database", "agi_assistant")

    # Elasticsearch
    if "elasticsearch" in data:
        es = data["elasticsearch"]
        cfg.storage.es.es_addresses = es.get("addresses", [])
        cfg.storage.es.es_username = es.get("username", "")
        cfg.storage.es.es_password = es.get("password", "")

    # Kafka
    if "kafka" in data:
        kf = data["kafka"]
        cfg.storage.kafka.kafka_brokers = kf.get("brokers", [])
        cfg.storage.kafka.kafka_topic = kf.get("topic", "agi-events")

    # Neo4j
    if "neo4j" in data:
        n4 = data["neo4j"]
        cfg.storage.neo4j.neo4j_uri = n4.get("uri", "")
        cfg.storage.neo4j.neo4j_user = n4.get("user", "")
        cfg.storage.neo4j.neo4j_password = n4.get("password", "")
        cfg.storage.neo4j.kg_max_hops = n4.get("max_hops", 2)
        cfg.storage.neo4j.kg_weight = n4.get("weight", 0.3)
        cfg.storage.neo4j.kg_enabled = n4.get("enabled", False)

    # RAG
    if "rag" in data:
        rag = data["rag"]
        cfg.rag.chunk_size = rag.get("chunk_size", 1000)
        cfg.rag.chunk_overlap = rag.get("chunk_overlap", 200)
        cfg.rag.top_k = rag.get("top_k", 5)
        cfg.rag.rrf_constant_k = rag.get("rrf_constant_k", 60)
        cfg.rag.semantic_weight = rag.get("semantic_weight", 0.7)
        cfg.rag.enable_hybrid_search = rag.get("enable_hybrid_search", True)
        cfg.rag.rag_milvus_dim = rag.get("rag_milvus_dim", 1024)
        if "rewrite" in rag:
            cfg.rag.rag_rewrite_enabled = rag["rewrite"].get("enabled", True)
            cfg.rag.rag_rewrite_num_queries = rag["rewrite"].get("num_queries", 3)
        if "rerank" in rag:
            cfg.rag.rag_rerank_enabled = rag["rerank"].get("enabled", True)
            cfg.rag.rag_rerank_preview_len = rag["rerank"].get("preview_len", 200)

    # Memory
    if "memory" in data:
        mem = data["memory"]
        cfg.memory.short_term_max_turns = mem.get("short_term_max_turns", 10)
        cfg.memory.long_term_top_k = mem.get("long_term_top_k", 5)
        if "consolidation" in mem:
            cons = mem["consolidation"]
            cfg.memory.memory_consolidation_similarity = cons.get("similarity_threshold", 0.80)
            cfg.memory.memory_consolidation_dedup = cons.get("dedup_threshold", 0.95)
            cfg.memory.memory_consolidation_ttl_days = cons.get("ttl_days", 30)
            cfg.memory.memory_consolidation_decay_rate = cons.get("decay_rate", 0.995)
            cfg.memory.memory_consolidation_min_import = cons.get("min_importance", 0.3)
            cfg.memory.memory_consolidation_trigger = cons.get("trigger_interval", 5)

    # Harness
    if "harness" in data:
        h = data["harness"]
        cfg.harness.max_retries = h.get("max_retries", 3)
        cfg.harness.retry_delay_ms = h.get("retry_delay_ms", 200)
        cfg.harness.step_timeout_ms = h.get("step_timeout_ms", 30000)
        cfg.harness.max_iterations = h.get("max_iterations", 10)

    # Search
    if "search" in data:
        s = data["search"]
        cfg.search.search_api_key = s.get("api_key", "")
        cfg.search.search_api_url = s.get("api_url", "")

    # Sandbox
    if "sandbox" in data:
        sb = data["sandbox"]
        cfg.sandbox.sandbox_enabled = sb.get("enabled", True)
        cfg.sandbox.sandbox_backend = sb.get("backend", "docker")
        cfg.sandbox.sandbox_image = sb.get("image", "ubuntu:22.04")
        cfg.sandbox.sandbox_timeout_ms = sb.get("timeout_ms", 30000)
        cfg.sandbox.sandbox_max_output = sb.get("max_output_bytes", 65536)
        cfg.sandbox.sandbox_memory_mb = sb.get("memory_limit_mb", 256)
        cfg.sandbox.sandbox_cpu_percent = sb.get("cpu_percent", 50)
        cfg.sandbox.sandbox_max_pids = sb.get("max_pids", 64)
        cfg.sandbox.sandbox_net_disabled = sb.get("network_disabled", True)
        cfg.sandbox.sandbox_read_only = sb.get("readonly_rootfs", True)

    # Security
    if "security" in data:
        sec = data["security"]
        cfg.security.sec_max_cmd_length = sec.get("max_command_length", 500)
        cfg.security.sec_allowlist_mode = sec.get("allowlist_mode", False)
        cfg.security.sec_allowlist = sec.get("allowlist", [])

    # Graph Runtime
    if "graph_runtime" in data:
        gr = data["graph_runtime"]
        cfg.graph_runtime.graph_max_parallel = gr.get("max_parallel", 2)
        cfg.graph_runtime.graph_race_timeout_ms = gr.get("race_timeout_ms", 30000)
        cfg.graph_runtime.graph_enable_racing = gr.get("enable_racing", True)

    _apply_defaults(cfg)
    return cfg
