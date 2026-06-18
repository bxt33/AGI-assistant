"""RAG 引擎：文本分块 + 三路混合检索 + LLM 合成

=============================================================================
                    📚 知识库检索的完整管道
=============================================================================

两个入口：
  ingest(text)  → 上传文档：切块 → Embedding → 三端写入（PG + Milvus + ES）
  query(text)   → 检索问答：改写 → 三路检索 → RRF融合 → 重排 → LLM合成

架构位置：
  core_agent._run_rag()
    └── self._rag.query(query, history)  ← 就是这里

设计理念：
  - 回调注入：通过 set_generate_fn / set_embed_fn 注入 LLM 函数，
    RAG 引擎本身不直接依赖 LLM 实现，测试时可以 Mock
  - 优雅降级：每一步都 try-catch，Milvus 挂了走 ES，ES 挂了走 TF 兜底
  - 三写一读：写入时三端全写（用空间换鲁棒性），查询时并行检索
  - 召回放大+精排：召回 top_k×3 候选 → LLM 精排 → 截回 top_k
=============================================================================
"""

import hashlib
import json
import logging
import threading
from typing import List, Optional, Callable, Dict

from src.domain.rag.splitter import split_text, Chunk
from src.domain.rag.hybrid import rrf_fusion
from src.domain.rag.rewriter import LLMRewriter, HistoryMessage
from src.domain.rag.reranker import LLMReranker

logger = logging.getLogger(__name__)


class Engine:
    """RAG 引擎主类

    成员变量速查：
      self._chunks   — 内存中的文档块列表 [Chunk(id=0, content="..."), ...]
      self._loaded   — 是否有已加载的文档（ingest 后或 restore 后变为 True）
      self._repo     — 多后端存储抽象层（封装了 PG + Milvus + ES 三端读写）
      self._events   — 事件发布器（Kafka / Log）
      self._mu       — 线程锁（保护 _chunks）
      self._generate_fn — LLM 生成回调（注入）
      self._embed_fn   — Embedding 回调（注入）
      self._rewriter   — LLM 查询改写器（可选）
      self._reranker   — LLM 结果重排器（可选）
    """

    def __init__(self, cfg, chunk_repo, event_publisher=None):
        self._cfg = cfg
        self._repo = chunk_repo  # ragchunk.Repo：PG + Milvus + ES 三后端的统一抽象
        self._events = event_publisher
        self._mu = threading.RLock()
        self._chunks: List[Chunk] = []
        self._loaded = False

        # 四个回调函数，由 core_agent._wire_rag_callbacks() 注入
        self._generate_fn: Optional[Callable[[str, str], str]] = None  # (system_prompt, user_query) -> answer
        self._embed_fn: Optional[Callable[[str], List[float]]] = None  # text -> [0.023, -0.451, ...]
        self._rewriter: Optional[LLMRewriter] = None                    # LLM 查询改写
        self._reranker: Optional[LLMReranker] = None                    # LLM 结果精排

    # ── 属性 ──
    @property
    def Loaded(self) -> bool:
        """（大写别名）兼容旧调用"""
        return self._loaded

    @property
    def loaded(self) -> bool:
        """是否有已加载的文档（_loaded 为 True 才能 query）"""
        return self._loaded

    # ── 回调注入（由 core_agent._wire_rag_callbacks() 调用）──
    def set_generate_fn(self, fn: Callable[[str, str], str]):
        """注入 LLM 生成函数：传入 system prompt + user query，返回回答文本"""
        self._generate_fn = fn

    def set_embed_fn(self, fn: Callable[[str], List[float]]):
        """注入 Embedding 函数：传入文本，返回向量"""
        self._embed_fn = fn

    def set_rewriter(self, rewriter: LLMRewriter):
        """注入查询改写器（可选，config.yaml: rag.rewrite.enabled）"""
        self._rewriter = rewriter

    def set_reranker(self, reranker: LLMReranker):
        """注入结果重排器（可选，config.yaml: rag.rerank.enabled）"""
        self._reranker = reranker

    # ── 状态查询 ──

    def mode(self) -> str:
        """
        返回当前检索模式描述（展示在启动横幅和 /api/status）。

        例：
          全部在线 → "milvus+es+tf"
          没 Docker → "tf"（只有 PG 内存兜底）
        """
        modes = []
        if self._repo and self._repo.MilvusAvailable():
            modes.append("milvus")
        if self._repo and self._repo.ESAvailable():
            modes.append("es")
        modes.append("tf")  # TF 兜底始终可用
        return "+".join(modes)

    def chunks(self) -> List[Chunk]:
        """返回内存中所有文档块（供前端展示用）"""
        with self._mu:
            return list(self._chunks)

    def restore_chunks(self, chunks: List[Chunk]):
        """
        从 PG 恢复到内存（服务重启后调用）。

        只在 _restore_rag_from_db() 中调用一次。
        直接替换 self._chunks，不做去重（PG 里已经是去重后的）。
        """
        with self._mu:
            self._chunks = chunks
            self._loaded = len(chunks) > 0

    # ═══════════════════════════════════════════════════════════════════════
    #  ingest() — 文档上传入口
    #
    #  完整流程：
    #    ① 计算 doc_hash（SHA256 前 16 位，用于后续删除）
    #    ② split_text() 滑动窗口切块
    #    ③ 每块生成 Embedding 向量
    #    ④ 三端写入：PG（原文）+ Milvus（向量）+ ES（倒排索引）
    #    ⑤ 更新内存 self._chunks
    #
    #  返回值：(chunk_count, doc_hash)
    # ═══════════════════════════════════════════════════════════════════════

    def ingest(self, text: str) -> tuple:
        """摄入文档到知识库

        参数：
          text — 文档原始文本（前端 FileReader 读出来的）

        返回：
          (saved_count, doc_hash) — 保存的块数 + 文档哈希（用于删除）

        注意：
          - 三端写入是串行的：先 PG（必须成功），再 Milvus，最后 ES
          - Milvus/ES 失败不影响 PG 写入（优雅降级）
          - 每块 ID 的计算方式：pg_id 由 PG 自增（RETURNING id）
        """
        # ① 计算文档哈希（用于前端"删除"按钮定位文档）
        doc_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        # ② 文本切块：滑动窗口，chunk_size=200字，overlap=50字
        #   例："Python是一种解释型语言，广泛用于AI开发..." (300字)
        #   → ["Python是一种解释型语言，广泛用于AI开发...",  (0~200)
        #      "广泛用于AI开发，特别是机器学习和深度学习..."]  (150~250，overlap部分)
        chunks = split_text(
            text,
            chunk_size=self._cfg.ChunkSize,       # 默认 200
            chunk_overlap=self._cfg.ChunkOverlap   # 默认 50
        )

        saved_count = 0
        for i, chunk_text in enumerate(chunks):
            # ③ 生成 Embedding 向量
            #    调 DeepSeek embedding API → [0.023, -0.451, 0.789, ...]（1536维）
            embedding = None
            embedding_json = None
            if self._embed_fn:
                try:
                    embedding = self._embed_fn(chunk_text)
                    embedding_json = json.dumps(embedding)  # 序列化为 JSON 字符串
                except Exception as e:
                    logger.warning(f"Embedding failed for chunk {i}: {e}")

            # ④-a 写入 PostgreSQL（数据源头，必须成功）
            try:
                pg_id, _ = self._repo.SavePG(doc_hash, i, chunk_text, embedding_json)
                if pg_id > 0:
                    saved_count += 1
            except Exception as e:
                logger.warning(f"Save PG failed for chunk {i}: {e}")
                continue  # PG 写失败就跳过当前块（不阻塞后续块）

        # ④-b 批量写入 Milvus（向量数据库，语义搜索用）
        #       失败不阻塞，打日志继续
        if embedding and self._repo.MilvusAvailable():
            try:
                pg_ids = []
                contents = []
                embeddings = []
                for i, chunk_text in enumerate(chunks):
                    emb = self._embed_fn(chunk_text) if self._embed_fn else None
                    if emb:
                        pg_ids.append(int(saved_count - len(chunks) + i + 1))
                        contents.append(chunk_text[:4096])  # Milvus 单条限制 4096 字符
                        embeddings.append([float(x) for x in emb])

                if pg_ids:
                    self._repo.InsertMilvus(pg_ids, contents, embeddings)
            except Exception as e:
                logger.warning(f"Milvus insert failed: {e}")

        # ④-c 索引到 Elasticsearch（全文搜索引擎，BM25 关键词匹配用）
        #       失败不阻塞
        if self._repo.ESAvailable():
            try:
                for i, chunk_text in enumerate(chunks):
                    self._repo.IndexES(
                        int(saved_count - len(chunks) + i + 1),
                        chunk_text, doc_hash, i
                    )
            except Exception as e:
                logger.warning(f"ES index failed: {e}")

        # ⑤ 更新内存索引（线程安全）
        #    之后 query() 就可以搜到这些块了
        with self._mu:
            self._chunks.extend(
                Chunk(id=len(self._chunks) + j, content=c)
                for j, c in enumerate(chunks)
            )
            self._loaded = True

        return saved_count, doc_hash

    def delete(self, doc_hash: str):
        """删除指定文档的所有 chunks（从 PG + Milvus + ES 三端同时删除）"""
        if self._repo:
            self._repo.Delete(doc_hash)

    # ═══════════════════════════════════════════════════════════════════════
    #  query() — 检索问答入口（这才是 RAG 的核心）
    #
    #  完整链路（6 步）：
    #    ① LLM 查询改写（消除指代歧义 + 生成多检索变体）
    #    ② 多路并行检索：Milvus（语义）+ ES（关键词）+ Neo4j（图谱）
    #    ③ RRF 倒数排名融合（去重 + 加权合并）
    #    ④ TF 兜底（Milvus 和 ES 都挂了时走纯内存）
    #    ⑤ LLM 重排（Listwise 精排，召回放大后截断）
    #    ⑥ LLM 合成答案（参考文档 + 用户问题 → 生成回答）
    #
    #  返回值：(answer_string, [ChunkRow, ...])
    # ═══════════════════════════════════════════════════════════════════════

    def query(self, query_text: str, history: Optional[List[HistoryMessage]] = None) -> tuple:
        """检索并合成回答

        参数：
          query_text — 用户原始问题
          history    — 最近 6 轮对话（用于查询改写时消除指代歧义）

        返回：
          (answer, chunks_loaded) — LLM 生成的答案 + 检索到的文档块列表
        """
        if not self._loaded:
            return "", []  # 还没上传过文档，直接返回空

        # ── ① LLM 查询改写 ──
        #    把口语化问题拆成多个检索变体，提高召回率
        #    例："它怎么用？" + history:["我刚传了AGI-Saber文档"]
        #    → ["AGI-Saber使用方法", "AGI-Saber入门指南", "AGI-Saber配置教程"]
        queries = [query_text]
        if self._rewriter:
            queries = self._rewriter.rewrite(query_text, history)

        # ── ② 多路并行检索 ──
        all_pg_ids = set()    # 收集所有候选的 PG ID
        milvus_hits = []      # [{pg_id, score}, ...]
        es_hits = []          # [{pg_id, score}, ...]
        kg_hits = []          # [{pg_id, score}, ...]

        for q in queries:
            # 生成查询向量（每个改写变体都要生成自己的 embedding）
            q_emb = None
            if self._embed_fn:
                try:
                    q_emb = self._embed_fn(q)
                except Exception:
                    pass

            # 路1：Milvus 向量语义搜索
            #      query_embedding → ANN 近似最近邻 → top_k×2 候选
            #      score = 1/(1+distance)，距离越近分数越高
            if q_emb and self._repo and self._repo.MilvusAvailable():
                try:
                    hits = self._repo.SearchMilvus(
                        [float(x) for x in q_emb], self._cfg.TopK * 2
                    )
                    milvus_hits.extend(
                        {"pg_id": h.ID, "score": 1.0 / (1.0 + h.Distance)}
                        for h in hits
                    )
                except Exception:
                    pass

            # 路2：Elasticsearch BM25 关键词搜索
            #      BM25：词频×逆文档频率，擅长精确匹配
            if self._repo and self._repo.ESAvailable():
                try:
                    hits = self._repo.SearchES(q, self._cfg.TopK)
                    es_hits.extend(
                        {"pg_id": h.PGID, "score": h.Score} for h in hits
                    )
                except Exception:
                    pass

        # ── ③ RRF 倒数排名融合 ──
        #     例：
        #       Milvus 排名: [A(1), C(2), B(3)]
        #       ES 排名:     [B(1), A(2), D(3)]
        #       RRF: A=1/(60+1)+1/(60+2), B=1/(60+3)+1/(60+1), ...
        #     最终按 RRF 分数排序 → [A, B, C, D]
        #     效果：两个引擎都认为重要的文档排前面
        if milvus_hits or es_hits or kg_hits:
            fused_ids = rrf_fusion(
                milvus_hits, es_hits, kg_hits,
                k=self._cfg.RRFConstantK,         # 默认 60（经典 RRF 参数）
                semantic_weight=self._cfg.SemanticWeight,   # 语义权重
                kg_weight=self._cfg.KGWeight,               # 图谱权重
            )
            all_pg_ids.update(fused_ids[:self._cfg.TopK])

        # ── ④ TF 兜底：Milvus 和 ES 都挂了时 ──
        #     直接取前 N 个 chunk（按内存顺序，不做语义匹配）
        if not all_pg_ids:
            all_pg_ids = set(range(min(self._cfg.TopK, len(self._chunks))))

        # ── ⑤ 从 PG 加载 chunk 原文 ──
        #     前几步只拿到 pg_id，这里真正去 PG 读原文内容
        chunks_loaded = []
        if all_pg_ids and self._repo:
            try:
                loaded = self._repo.LoadByIDs(list(all_pg_ids)[:self._cfg.TopK * 3])
                chunks_loaded = loaded
            except Exception:
                pass

        # ── ⑥ LLM 重排（Listwise Rerank）──
        #     召回 top_k×3 篇 → LLM 按相关性重新排序 → 截回 top_k
        #     为什么：向量相似 ≠ 真正相关，LLM 能理解语义相关度
        chunk_contents = [c.Content for c in chunks_loaded] if chunks_loaded else []
        if self._reranker and len(chunk_contents) > 1:
            order = self._reranker.rerank(query_text, chunk_contents)
            chunks_loaded = [chunks_loaded[i] for i in order[:self._cfg.TopK]]

        # ── ⑦ LLM 合成最终答案 ──
        #     把检索到的文档块作为参考文档，让 LLM 据此回答
        if chunks_loaded and self._generate_fn:
            context = "\n\n---\n".join(
                c.Content[:1500] for c in chunks_loaded[:self._cfg.TopK]
            )
            system = f"""你是一个基于知识的问答助手。请根据以下参考文档回答用户问题。
如果参考文档中没有相关信息，请如实告知。

参考文档：
{context}"""
            answer = self._generate_fn(system, query_text)
            return answer, chunks_loaded

        # 没找到相关文档 → 返回空答案 + 加载到的块（可能为空）
        return "", chunks_loaded
