"""UnifiedAgent：整合全部能力的核心调度器。

=============================================================================
                       🧠 这是整个后端最重要的文件
=============================================================================

一句话概括：收到用户消息 → 选模式 → 拼提示词 → 调 LLM/工具/RAG → 返回答案

架构位置（在整体链路中的角色）：
  handler.py 的 chat() → agent.Process(message, opts) → 就是这里！

完整的请求生命周期：
  Process(message, opts)
    │
    ├── ① STM.Add(user, message)          存短期记忆
    ├── ② _route_mode(query, opts)        路由决策：chat / tool / rag / react
    ├── ③ build_context_prefix(query)      拼系统提示词（Schema 驱动）
    ├── ④ build_history_messages(query)    取最近 N 轮对话
    ├── ⑤ 按模式执行（四选一）:
    │       _run_chat()   → 直接调 LLM
    │       _run_tool()   → 规则匹配工具 → 执行 → LLM 合成
    │       _run_rag()    → 查知识库 → LLM 合成
    │       _run_react()  → Planner 规划 → Executor 循环执行 → Generator 生成答案
    ├── ⑥ STM.Add(assistant, answer)      存 AI 回复到短期记忆
    ├── ⑦ PG 持久化聊天记录
    ├── ⑧ 异步提取用户偏好（LLM + 正则）
    └── ⑨ 异步存储长期记忆（降级 + 合并检查）

四种模式与 config.yaml 开关的关系：
  - 纯聊天 Chat：默认兜底，什么都不开就走这个
  - 知识库 RAG：前端"知识库"开关打开 + 已上传文档 → 检索后回答
  - 单工具 Tool：检测到"天气/时间/搜索"等单关键词 → 调用一个工具
  - 多步推理 ReAct：检测到 2+ 关键词 或 用户手动选了工具 → LLM 规划多步执行

路由策略（按优先级，_route_mode 方法）：
 1. explicit=True → 用户手动选了工具 → ReAct（多工具）或 RAG
 2. _need_react → 查询含 2+ 关键词 → ReAct
 3. _need_tool  → 查询含单个关键词 → Tool
 4. use_rag + loaded → RAG
 5. 都不满足 → Chat（兜底）
=============================================================================
"""

import json
import logging
import threading
import time
import traceback
from typing import Optional, Callable, List, Dict

from src.application.chat.core_types import (
    ChatOptions, Response, StreamEvent, TaskState, TaskStep, Snapshot,
    TaskStepStatus, ReActStep, StepType,
)
from src.application.chat.ctx_builder import (
    build_context_prefix, build_system_prompt, build_history_messages,
    recent_history_for_rag, filter_tools, emit, chat_llm,
)
from src.application.chat.ctx_prompt import PromptCtx
from src.application.chat.infra_repos import RepoBundle
from src.application.chat.mem_stack import MemoryStack
from src.application.chat.runtime_task import TaskRuntime
from src.application.chat.tool_registry import ToolRegistry

from src.domain.memory.longterm import Item as LTMItem, RecallFilter
from src.domain.memory.graphmem import GraphMemory
from src.domain.promptctx.assembler import SourceRegistry
from src.domain.promptctx.source_profile import ProfileSource
from src.domain.promptctx.source_planner import PlannerSource
from src.domain.promptctx.source_taskmem import TaskMemSource
from src.domain.promptctx.source_tools import ToolStateSource
from src.domain.promptctx.source_constraints import ConstraintsSource
from src.domain.promptctx.source_recall import RecallSource
from src.domain.promptctx.schema import default_schemas
from src.domain.promptctx.assembler import ContextAssembler
from src.domain.promptctx.source_tools import ToolCallTrace
from src.domain.promptctx.source_taskmem import StepObservation
from src.domain.sandbox.validator import policy_snapshot
from src.domain.tool import Tool, Param, decide
from src.domain.rag.engine import Engine as RAGEngine
from src.domain.rag.rewriter import LLMRewriter
from src.domain.rag.reranker import LLMReranker
from src.domain.knowledge.kgstore import KGStore
from src.domain.knowledge.extractor import Extractor
from src.domain.knowledge.types import ChunkRef
from src.domain.sandbox.types import SandboxConfig, SecurityConfig

from src.infrastructure.llm import Client as LLMClient, Message
from src.infrastructure.sandbox.factory import new_sandbox
from src.infrastructure.tool.builtin import default_tools
from src.infrastructure.tool.exec_command import exec_command_tool
from src.infrastructure.tool.tavily import tavily_search

logger = logging.getLogger(__name__)

# 兜底系统提示词：当用户什么都没配置时使用
_CHAT_PROMPT = """你是一个全能的 AGI 智能助手，具备丰富的知识和推理能力。
请用简洁清晰的中文回答用户问题。如果用户询问你的能力，请如实介绍你有对话、RAG知识库检索、工具调用、多步推理等能力。"""


class UnifiedAgent:
    """整合全部能力，是系统的核心调度入口

    成员变量速查（初始化完成后）：
      self._cfg      — 配置对象（读自 config.yaml）
      self._llm      — LLM 客户端（DeepSeek / OpenAI 兼容）
      self._rag      — RAG 引擎（文档分块 → Embedding → 三路检索 → LLM 合成）
      self._tools    — 工具注册表（时间/天气/搜索/命令执行/MCP）
      self._runtime  — 任务运行时（ReAct 模式下追踪当前任务状态）
      self._mem      — 记忆栈（STM + LTM + 偏好 + 图增强记忆）
      self._repos    — 仓库包（PG 持久化的 5 个 Repository）
      self._kg       — 知识图谱（Neo4j，可选）
      self._pctx     — 提示词上下文装配器（Schema 驱动，6 个 Source）
      self._sandbox  — 沙箱执行器（Docker/Local/Mock）
    """

    def __init__(self, cfg, deps: dict):
        """
        参数：
          cfg  — APIConfig，读自 config.yaml
          deps — 依赖注入字典（main.py 中组装好后传入）
                包含：chat, pref, snap, ltm, rag_chunk, events, infra_status

        构造流程（按顺序）：
          1. 创建 LLM 客户端
          2. 创建 RAG 引擎
          3. 创建工具注册表（先注册默认工具：time/weather/search_web）
          4. 创建任务运行时
          5. 创建记忆栈（STM + LTM + 偏好）
          6. 创建仓库包（5 个 PG Repository + EventPublisher）
          7. 注入 RAG 回调（LLM 调用 / Embedding / 改写 / 重排）
          8. 异步引导初始化 _bootstrap()
        """
        self._cfg = cfg
        # LLM 客户端：封装 DeepSeek API（兼容 OpenAI 格式）
        # 支持 chat()（同步）、chat_stream()（流式）、embed()（向量化）、extract_preferences()（提取偏好）
        self._llm = LLMClient(cfg)

        # RAG 引擎：文档分块 → Embedding → 三路检索（Milvus+ES+Neo4j）→ RRF 融合 → LLM 合成
        # rag_chunk 是多后端存储（PG 存原文 + Milvus 存向量 + ES 存索引）
        self._rag = RAGEngine(cfg, deps.get("rag_chunk"), deps.get("events"))

        # 工具注册表：Map<name, Tool>，线程安全
        # 默认注册 3 个内置工具：get_time, get_weather, search_web
        self._tools = ToolRegistry(default_tools())

        # 任务运行时：ReAct 模式下追踪当前任务和执行状态
        self._runtime = TaskRuntime()

        # 记忆栈：聚合 STM + LTM + Preference + GraphMemory
        self._mem = MemoryStack(cfg)

        # 仓库包：5 个 PG Repository（聊天/偏好/快照/长期记忆/RAG块）+ EventPublisher
        self._repos = RepoBundle(
            chat=deps.get("chat"),
            pref=deps.get("pref"),
            snap=deps.get("snap"),
            ltm=deps.get("ltm"),
            rag_chunk=deps.get("rag_chunk"),
            events=deps.get("events"),
            infra_status=deps.get("infra_status", {})
        )

        # 知识图谱：Neo4j，可选（连接失败时为 None）
        self._kg: Optional[KGStore] = None

        # 提示词上下文装配器：Schema 驱动，6 个 Source，并发填充 Slot
        self._pctx: Optional[PromptCtx] = None

        # ── 注入回调 + 异步初始化 ──
        self._wire_rag_callbacks()  # 把 LLM/Embedding 函数注入 RAG 引擎
        self._bootstrap()           # 异步：初始化基础设施 + 恢复数据 + 构建提示词装配器

    # ═══════════════════════════════════════════════════════════════════════
    #  启动初始化
    # ═══════════════════════════════════════════════════════════════════════

    def _wire_rag_callbacks(self):
        """
        把 LLM 调用 / Embedding 生成 / 查询改写 / 结果重排 四个函数注入 RAG 引擎。

        设计理念：RAG 引擎本身是纯领域逻辑（domain/rag/engine.py），不直接依赖 LLM。
        通过回调注入，可以方便地替换 LLM 实现（如测试时用 Mock）。
        """
        cfg = self._cfg

        # 基础回调：LLM 生成 + Embedding 向量化
        self._rag.set_generate_fn(lambda system, user: self._llm.chat(system, [Message(role="user", content=user)]))
        self._rag.set_embed_fn(lambda text: self._llm.embed(text))

        # 查询改写（可选，config.yaml: rag.rewrite.enabled）：
        # 把用户口语化问题拆成多个检索变体，提高召回率
        # 如 "数据库为啥选PG" → ["PostgreSQL选型原因", "数据库对比 PG MySQL", "为什么选择PostgreSQL"]
        if cfg.RAGRewriteEnabled and cfg.RAGRewriteNumQueries > 1:
            rewrite_llm = lambda s, u: self._llm.chat(s, [Message(role="user", content=u)])
            self._rag.set_rewriter(LLMRewriter(rewrite_llm, cfg.RAGRewriteNumQueries))

        # 结果重排（可选，config.yaml: rag.rerank.enabled）：
        # 检索阶段召回 4×top_k 个候选，用 LLM 重新排序后截回 top_k
        if cfg.RAGRerankEnabled:
            rerank_llm = lambda s, u: self._llm.chat(s, [Message(role="user", content=u)])
            self._rag.set_reranker(LLMReranker(rerank_llm, cfg.RAGRerankPreviewLen))

    def _bootstrap(self):
        """
        启动期初始化（顺序执行，但每个子步骤内部有异步保护）：
          1. 初始化 RAG 基础设施（建 Milvus 集合 + ES 索引）
          2. 从 PG 恢复偏好 / 长期记忆 / 聊天记录
          3. 从 PG 恢复 RAG 文档块
          4. 初始化沙箱（docker/local/mock）+ 注册 exec_command 工具
          5. 注册内置工具（rag_search + search_web）
          6. 初始化知识图谱（Neo4j，可选）
          7. 构建提示词装配器（6 个 Source 注册 → Schema → ContextAssembler）
        """
        # 初始化 RAG 基础设施（异步线程，失败不阻塞）
        self._go_safe("InitRAGInfra", lambda: self._repos.rag_chunk.Init(self._cfg.RAGMilvusDim))
        # 恢复持久化数据
        self._restore_from_db()
        self._restore_rag_from_db()
        # 初始化沙箱
        self._init_sandbox()
        # 注册内置工具
        self._register_builtin_tools()
        # 初始化知识图谱
        self._init_knowledge_graph()
        # 构建 prompt 装配器
        self._build_prompt_ctx()

    def _go_safe(self, name: str, fn: Callable[[], None]):
        """
        在 daemon 线程中执行函数，异常不崩溃，只打日志。

        用途：异步保存偏好/记忆/快照等非关键操作。
        为什么用 daemon=True？主进程退出时这些线程自动回收，不阻塞关闭。
        """
        def wrapper():
            try:
                fn()
            except Exception as e:
                logger.error(f"Thread error [{name}]: {e}\n{traceback.format_exc()}")
        t = threading.Thread(target=wrapper, daemon=True)
        t.start()

    def _restore_from_db(self):
        """
        从 PostgreSQL 恢复跨会话数据（服务重启后仍然保留）：
          - 用户偏好（"我叫张三""喜欢简洁回答"）
          - 长期记忆（过去对话中提取的重要信息）
          - 聊天记录（最近 N 轮对话，N = ShortTermMaxTurns × 2）
        """
        try:
            # 恢复偏好 → Preference 内存
            prefs = self._repos.pref.load("default")
            self._mem.pref.save_batch(prefs)

            # 恢复长期记忆 → LTM 内存（含 embedding 向量）
            rows = self._repos.ltm.load()
            for row in rows:
                self._mem.ltm.store_item(LTMItem(
                    id=row.id, content=row.content, importance=row.importance,
                    embedding=row.embedding, created_at=row.created_at,
                    last_accessed=row.last_accessed, category=row.category,
                    tags=row.tags, slot_hint=row.slot_hint,
                ))

            # 恢复聊天记录 → STM（最近 N 轮，旧的自动淘汰）
            chat_limit = self._cfg.ShortTermMaxTurns * 2
            history = self._repos.chat.load(chat_limit)
            for h in history:
                self._mem.stm.add(h.role, h.content)

            if prefs or rows or history:
                logger.info(f"记忆恢复：{len(prefs)} 条偏好，{len(rows)} 条长期记忆，{len(history)} 条聊天记录")
        except Exception as e:
            logger.warning(f"restoreFromDB failed: {e}")

    def _restore_rag_from_db(self):
        """从 PostgreSQL 恢复上次上传的知识库文档块"""
        try:
            chunk_rows = self._repos.rag_chunk.LoadAll()
            if chunk_rows:
                from src.domain.rag.splitter import Chunk
                chunks = [Chunk(id=i, content=r.content) for i, r in enumerate(chunk_rows)]
                self._rag.restore_chunks(chunks)
                logger.info(f"RAG chunks 恢复：{len(chunks)} 条")
        except Exception as e:
            logger.warning(f"restoreRAGFromDB failed: {e}")

    def _init_sandbox(self):
        """
        初始化沙箱执行器 + 注册 exec_command 工具。

        沙箱三种后端：
          docker  — 用 docker run 隔离执行（需要 Docker Desktop）
          local   — 子进程直接执行（只允许 SAFE 级别命令）
          mock    — 返回假数据（Docker 不可用时自动降级）
        """
        sb_cfg = SandboxConfig(
            image=self._cfg.SandboxImage,
            timeout=self._cfg.SandboxTimeoutMs / 1000.0,
            max_output_bytes=self._cfg.SandboxMaxOutput,
            memory_limit_mb=self._cfg.SandboxMemoryMB,
            cpu_percent=self._cfg.SandboxCPUPercent,
            max_pids=self._cfg.SandboxMaxPIDs,
            network_disabled=self._cfg.SandboxNetDisabled,
            read_only_rootfs=self._cfg.SandboxReadOnly,
        )
        sec_cfg = SecurityConfig(
            max_command_length=self._cfg.SecMaxCmdLength,
            allowlist_mode=self._cfg.SecAllowlistMode,
            allowlist=self._cfg.SecAllowlist,
        )
        self._sandbox = new_sandbox(self._cfg.SandboxBackend, sb_cfg, sec_cfg)
        # 注册 exec_command 工具（名称: exec_command，沙箱隔离执行）
        self._tools.register(exec_command_tool(self._sandbox))

    def _register_builtin_tools(self):
        """
        注册 RAG 检索工具 + 网络搜索工具。

        rag_search  — 从知识库中检索（调用 RAG 引擎的 query）
        search_web  — 搜索互联网（Tavily API 优先，降级为 LLM 生成）
        """
        # rag_search：把 RAG 引擎包装成 Tool 接口
        self._tools.register(Tool(
            name="rag_search",
            description="从私人黑洞（个人知识库）中检索相关文档内容",
            parameters=[Param(name="query", type="string", description="检索关键词或问题", required=True)],
            execute=lambda p: self._rag_query(p.get("query", "")),
        ))
        # search_web：Tavily API 优先，降级 LLM 兜底
        self._tools.register(Tool(
            name="search_web",
            description="搜索互联网获取最新信息",
            parameters=[Param(name="query", type="string", description="搜索关键词", required=True)],
            execute=lambda p: self._search_web_exec(p.get("query", "")),
        ))

    def _rag_query(self, q: str) -> str:
        """RAG 工具的执行体：调 RAG 引擎检索 + 生成"""
        if not self._rag.loaded:
            return "知识库为空，请先在「私人黑洞」上传文档"
        answer, __ = self._rag.query(q)
        return answer

    def _search_web_exec(self, q: str) -> str:
        """
        网络搜索的执行体：
          优先 → Tavily API（需配置 search.api_key）
          降级 → LLM 用自身知识生成回答
        """
        if not q:
            return "搜索关键词不能为空"
        if self._cfg.SearchAPIKey:
            try:
                result = tavily_search(q, self._cfg.SearchAPIKey, self._cfg.SearchAPIURL)
                if result and "失败" not in result:
                    return result
            except Exception:
                pass
        # 降级：LLM 假装搜索（用自身训练数据回答）
        return self._llm.chat(
            "你是一个知识丰富的搜索引擎助手。请基于你的知识，对用户的搜索问题给出准确、详细的回答。",
            [Message(role="user", content="搜索：" + q)]
        )

    def _init_knowledge_graph(self):
        """
        初始化 Neo4j 知识图谱（可选，config.yaml: neo4j.enabled）。

        作用：
          - 从文档中抽取实体和关系（LLM 抽取）
          - 在 RAG 检索中增加图检索维度（三路混合检索的第三路）
          - 长期记忆叠加图关系（FOLLOWS/SIMILAR_TO/CAUSES/BELONGS_TO）
        """
        from src.infrastructure.platform.neo4j import connect as neo4j_connect
        neo4j_client = neo4j_connect(self._cfg.Neo4jConfig)
        if neo4j_client and neo4j_client.available:
            extractor = Extractor(lambda s, u: self._llm.chat(s, [Message(role="user", content=u)]))
            self._kg = KGStore(neo4j_client, self._cfg.KGMaxHops,
                               self._cfg.KGWeight, extractor)
            # 用 Neo4j 图增强记忆恢复（替代纯 LTM 语义搜索）
            self._mem.graph_mem = GraphMemory(neo4j_client, self._mem.ltm)
            # 将 KGStore 注入 RAG 引擎：ingest 时写实体关系，query 时图检索
            self._rag.set_kg_store(self._kg)

    def _build_prompt_ctx(self):
        """
        构建 Schema 驱动的提示词装配器。

        这是整个系统最精妙的设计之一：
          不是在代码里写死一个超长系统提示词，
          而是根据当前模式（chat/tool/react/rag），
          动态选择 Schema → 并发填充 6 个 Slot → 按预算裁剪 → 渲染中文提示词。

        6 个 Source（数据来源）：
          ProfileSource       → 【用户画像】偏好 + 身份
          PlannerSource       → 【任务规划】当前 ReAct 任务状态
          TaskMemSource       → 【任务记忆】最近步骤的观察结果
          ToolStateSource     → 【工具状态】可用工具列表 + 调用痕迹
          ConstraintsSource   → 【硬性约束】沙箱安全策略
          RecallSource        → 【相关回忆】LTM 或 GraphMemory 语义召回

        工作流程（每次请求时触发）：
          Schema（按 mode 选）→ 并发填充 Slot（每个 Slot 调对应 Source）
          → 收集 FilledSlot → 按优先级裁剪（超预算时删低优先级）
          → RuntimeContext.render() → 中文提示词 → 拼到系统 prompt 前面
        """
        self._pctx = PromptCtx()

        reg = SourceRegistry()
        reg.register(ProfileSource(self._mem.pref, self._mem.ltm))
        reg.register(PlannerSource(lambda: self._planner_snapshot()))
        reg.register(TaskMemSource(self._pctx.task_mem))
        reg.register(ToolStateSource(lambda: self._tools.snapshot(), self._pctx.tool_tracker))
        reg.register(ConstraintsSource(policy_snapshot()))
        # 图记忆可用时用 GraphMemory，否则降级到纯 LTM
        if self._mem.graph_mem and self._mem.graph_mem.available():
            reg.register(RecallSource(self._mem.graph_mem))
        else:
            reg.register(RecallSource(self._mem.ltm))

        self._pctx.assembler = ContextAssembler(default_schemas(), reg)

    def _planner_snapshot(self):
        """获取当前 ReAct 任务的快照（用于填充 PLANNER Slot）"""
        t = self._runtime.current_task()
        if not t:
            return None
        from src.domain.promptctx.source_planner import PlannerSnapshot
        snap = PlannerSnapshot(
            task_id=t.task_id, query=t.query, status=t.status, phase=t.phase,
            total_steps=len(t.steps), current_step=t.current_step,
            interrupted_at=t.interrupted_at,
        )
        if t.current_step + 1 < len(t.steps):
            nxt = t.steps[t.current_step + 1]
            snap.next_step_name = nxt.name
            snap.next_step_tool = nxt.tool_name
        return snap

    # ═══════════════════════════════════════════════════════════════════════
    #  访问器（供 handler.py 和前端调用）
    # ═══════════════════════════════════════════════════════════════════════

    def RegisterTool(self, t: Tool):
        """动态注册工具（MCP 工具通过此方法运行时注册）"""
        self._tools.register(t)

    def RAG(self): return self._rag
    def Tools(self): return self._tools.snapshot()
    def ShortTerm(self): return self._mem.stm
    def LongTerm(self): return self._mem.ltm
    def Preferences(self): return self._mem.pref
    def Snapshots(self): return self._runtime.snapshot_list()
    def Cancel(self): self._runtime.cancel_all()
    def current_task(self): return self._runtime.current_task()
    def InfraStatus(self): return self._repos.infra_snapshot()

    def Status(self) -> dict:
        """返回系统健康状态（GET /api/status 用）"""
        chunk_previews = []
        for c in self._rag.chunks():
            preview = c.content[:60] + "..." if len(c.content) > 60 else c.content
            chunk_previews.append({"id": c.id, "content": preview})
        return {
            "rag_loaded": self._rag.loaded,
            "rag_mode": self._rag.mode(),
            "rag_chunks": chunk_previews,
            "short_term_count": self._mem.stm.count(),
            "long_term_count": self._mem.ltm.count(),
            "preferences": self._mem.pref.snapshot(),
            "tools_count": len(self._tools.snapshot()),
            "llm_model": self._cfg.LLMModel,
            "embedding_model": self._cfg.EmbeddingModel,
            "is_mock": not self._cfg.IsRealLLM,
            "infrastructure": self.InfraStatus(),
        }

    # ═══════════════════════════════════════════════════════════════════════
    #  路由判断（_route_mode）
    #  ★ 决定走哪种模式的核心逻辑 ★
    # ═══════════════════════════════════════════════════════════════════════

    def _need_tool(self, query: str) -> bool:
        """
        判断是否为单工具触发场景。

        规则：查询中包含任一工具关键词
          时间相关："几点"、"时间"
          天气相关："天气"
          搜索相关："查"、"搜索"、"是什么"

        满足 → 走 Tool 模式（_run_tool）
        """
        q = query.lower()
        return any(kw in q for kw in ["几点", "时间", "天气", "查", "搜索", "是什么"])

    def _need_react(self, query: str) -> bool:
        """
        判断是否为多步推理场景。

        规则：查询中同时包含 2 个或以上的工具关键词
          如 "北京今天天气怎么样？顺便查一下最近有什么新闻" → 2 个关键词 → ReAct

        满足 → 走 ReAct 模式（_run_react）：LLM 规划 → 多步执行 → 合成答案
        """
        q = query.lower()
        count = sum(1 for kw in ["时间", "几点", "天气", "总结", "汇总", "查", "搜索"] if kw in q)
        return count >= 2

    def _route_mode(self, query: str, opts: ChatOptions) -> str:
        """
        路由决策核心（按优先级从高到低）：

        优先级 1 — explicit 模式（用户手动指定）：
          如果前端传了 explicit=true 且选了工具 → ReAct（多工具编排）
          如果前端传了 explicit=true 且开了知识库 → RAG（查知识库）
          如果前端传了 explicit=true 但什么都没开 → Chat（兜底）

        优先级 2 — ReAct（多步推理）：
          查询含 2+ 关键词（如"查天气并总结"）→ ReAct

        优先级 3 — Tool（单工具）：
          查询含 1 个关键词（如"现在几点"）→ Tool

        优先级 4 — RAG（知识库）：
          前端开了知识库开关 + 已上传文档 → RAG

        优先级 5 — Chat（兜底）：
          以上都不满足 → 直接和 LLM 对话

        返回值："chat" | "tool" | "rag" | "react"
        """
        # 优先级 1：用户显式指定模式
        if opts.explicit:
            ts = filter_tools(self, opts.selected_tools or [])
            if len(ts) > 0:
                return "react"           # 有选中工具 → ReAct 多步执行
            if opts.use_rag:
                return "rag" if self._rag.loaded else "chat"  # 有知识库 → RAG
            return "chat"               # 都没选 → 纯聊天

        # 优先级 2：多关键词 → ReAct
        if self._need_react(query):
            return "react"

        # 优先级 3：单关键词 → Tool
        if self._need_tool(query):
            return "tool"

        # 优先级 4：知识库开关开 + 已加载 → RAG
        if opts.use_rag and self._rag.loaded:
            return "rag"

        # 优先级 5：兜底 → Chat
        return "chat"

    # ═══════════════════════════════════════════════════════════════════════
    #  主入口 Process() — 整个系统的"主函数"
    #  ★★★★★ 这是最重要的方法 ★★★★★
    #
    #  每当用户发一条消息，无论是同步 /api/chat 还是流式 /api/chat/stream，
    #  最终都会进入这个方法（或其流式版本 ProcessStream）。
    #
    #  阅读顺序（按行号）：
    #    L1 → stm.add("user")       存用户的原始消息
    #    L2 → _route_mode()         路由：chat/tool/rag/react 四选一
    #    L3 → build_context_prefix() 拼系统提示词（偏好+回忆+约束）
    #    L4 → build_history_messages() 取最近 N 轮历史对话
    #    L5 → _run_xxx()            按模式执行（调 LLM / 工具 / RAG / ReAct）
    #    L6 → stm.add("assistant")  存 AI 回复
    #    L7 → PG 持久化聊天记录
    #    L8 → 异步提取偏好（_go_safe）
    #    L9 → 异步存长期记忆（_go_safe）
    #    L10 → 返回 Response
    # ═══════════════════════════════════════════════════════════════════════

    def Process(self, query: str, opts: ChatOptions = None) -> Response:
        """
        同步处理一条用户消息（POST /api/chat）。

        参数：
          query — 用户输入的文本
          opts  — 前端传来的配置（知识库开关、选中的工具列表、是否显式模式）

        返回：
          Response 对象，包含 answer/mode/steps/preferences 等，被 handler.py 序列化为 JSON

        注意：LLM 调用是同步阻塞的，如需流式请用 ProcessStream()
        """
        opts = opts or ChatOptions()

        # ── 步骤 1：存短期记忆（用户消息）──
        # STM 是滑动窗口，超过 MaxTurns×2 条后自动淘汰最旧的
        self._mem.stm.add("user", query)

        # ── 步骤 2：路由决策 ──
        # 根据 query 内容 + opts 配置，决定走 chat/tool/rag/react
        mode = self._route_mode(query, opts)
        emit(None, "route", {"mode": mode})

        # ── 步骤 3：拼系统提示词前缀 ──
        # Schema 驱动：按 mode 选 Schema → 并发填充 Slot → 渲染中文提示词
        # 产物示例：
        #   【硬性约束】- [禁止] 禁止删除根路径
        #   【用户画像】- 姓名: 张三 | 喜好: 编程
        #   【相关回忆】- 上次问了Python异步编程问题...
        mem_prefix = build_context_prefix(self, query, mode)

        # ── 步骤 4：取历史对话 ──
        # 从 STM 滑动窗口取出最近 N 轮对话，拼成 Message 列表
        history = build_history_messages(self, query)

        # ── 步骤 5：按模式执行（四个分支）──
        if mode == "react":
            # ReAct：Planner 规划 → Executor 循环执行 → Generator 生成答案
            answer, steps = self._run_react(query, mem_prefix, history, opts)
        elif mode == "tool":
            # Tool：规则匹配工具（decide）→ 执行 → LLM 合成答案
            answer, steps = self._run_tool(query, mem_prefix, history)
        elif mode == "rag":
            # RAG：改写查询 → 三路检索 → RRF 融合 → LLM 合成答案
            answer, steps = self._run_rag(query)
        else:
            # Chat：系统提示词 + 历史 + 用户问题 → 直接调 LLM
            answer, steps = self._run_chat(query, mem_prefix, history)

        # ── 步骤 6：存 AI 回复到短期记忆 ──
        self._mem.stm.add("assistant", answer)

        # ── 步骤 7：持久化聊天记录到 PostgreSQL ──
        # 每条对话都存两条：用户消息 + 助手回复
        self._repos.chat.save("user", query)
        self._repos.chat.save("assistant", answer)

        # ── 步骤 8：异步提取用户偏好（LLM + 正则）──
        # 在 daemon 线程中执行，不阻塞响应返回
        # 如用户说"我喜欢简洁的回答" → 提取 preference: 回答风格=简洁
        self._go_safe("extractPrefs", lambda: self._extract_and_save_prefs(query))

        # ── 步骤 9：异步存储长期记忆 ──
        # 生成 Embedding → 分类存储 → 去重检查 → 合并检查
        # 图记忆可用时还会在 Neo4j 中建立 FOLLOWS 关系
        self._go_safe("storeLTM", lambda: self._store_long_term_memory(query))

        # ── 步骤 10：返回响应 ──
        # 注意：步骤 8/9 是异步的，此时可能还没完成
        return Response(
            query=query, answer=answer, mode=mode, steps=steps,
            short_term_count=self._mem.stm.count(),
            long_term_count=self._mem.ltm.count(),
            preferences=self._mem.pref.snapshot(),
        )

    def ProcessStream(self, query: str, opts: ChatOptions,
                      on_event: Callable[[StreamEvent], None]):
        """
        流式处理一条用户消息（POST /api/chat/stream）。

        与 Process() 的区别：
          - 同步版等全部完成后一次性返回 Response
          - 流式版每产生一个 token/事件就通过 on_event 回调推送给前端

        on_event 回调的类型（前端 SSE 事件）：
          route      — 路由决策结果
          step       — ReAct 推理步骤（思考→动作→观察）
          tool_call  — 工具调用详情
          rag_result — RAG 检索结果
          memory     — 记忆提取信息
          token      — LLM 输出的文字片段（打字机效果）
          done       — 处理完成，含最终 answer/mode/steps
        """
        opts = opts or ChatOptions()
        self._mem.stm.add("user", query)
        mode = self._route_mode(query, opts)
        on_event(StreamEvent(type="route", data={"mode": mode}))

        mem_prefix = build_context_prefix(self, query, mode)
        history = build_history_messages(self, query)

        # 四模式分发（每个模式的 on_event 参数用于流式输出 token）
        if mode == "react":
            answer, steps = self._run_react(query, mem_prefix, history, opts, on_event)
        elif mode == "tool":
            answer, steps = self._run_tool(query, mem_prefix, history, on_event)
        elif mode == "rag":
            answer, steps = self._run_rag(query, on_event)
        else:
            answer, steps = self._run_chat(query, mem_prefix, history, on_event)

        # 后处理（与同步版相同）
        self._mem.stm.add("assistant", answer)
        self._repos.chat.save("user", query)
        self._repos.chat.save("assistant", answer)

        self._go_safe("extractPrefs", lambda: self._extract_and_save_prefs(query))
        self._go_safe("storeLTM", lambda: self._store_long_term_memory(query))

        # 发送 done 事件（前端收到后停止加载动画）
        on_event(StreamEvent(type="done", data={
            "answer": answer, "mode": mode, "steps": [s.__dict__ for s in steps],
        }))

    # ═══════════════════════════════════════════════════════════════════════
    #  四种模式实现
    # ═══════════════════════════════════════════════════════════════════════

    # ── 模式 1：Chat（纯对话）──
    # 最简模式：系统提示词 + 历史消息 → LLM → 回答
    def _run_chat(self, query: str, mem_prefix: str, history: List[Message],
                  on_event: Callable = None) -> tuple:
        """纯对话模式：直接调 LLM"""
        system = build_system_prompt(mem_prefix, _CHAT_PROMPT)
        # chat_llm: 调 LLM 并处理流式输出
        #   同步版 on_event=None → 等全部返回
        #   流式版 on_event 非空 → 每 token 回调一次
        answer = chat_llm(self, system, history, on_event)
        return answer, []

    # ── 模式 2：Tool（单工具调用）──
    # 流程：decide() 规则匹配 → 执行工具 → LLM 用工具结果合成答案
    def _run_tool(self, query: str, mem_prefix: str, history: List[Message],
                  on_event: Callable = None) -> tuple:
        """
        单工具调用模式。

        decide() 做的事情（见 src/domain/tool.py）：
          关键词匹配 → 选工具 + 提取参数
          "北京天气" → tool=get_weather, params={city:"北京"}
          "现在几点" → tool=get_time, params={}
          "搜索量子计算" → tool=search_web, params={query:"量子计算"}

        如果 decide() 返回 None（没匹配到任何工具），降级为 Chat 模式。
        """
        tools = self._tools.snapshot()
        tc = decide(query, tools)  # 规则匹配：关键词 → 工具 + 参数
        if tc is None:
            # 没匹配到工具 → 降级为纯聊天
            answer = chat_llm(self, build_system_prompt(mem_prefix, _CHAT_PROMPT), history, on_event)
            return answer, []

        # 通知前端：正在调用工具
        emit(on_event, "tool_call", {"tool_name": tc.tool_name, "params": tc.params})

        # 执行工具（get_time / get_weather / search_web / exec_command / ...）
        tool = tools.get(tc.tool_name)
        if tool and tool.execute:
            try:
                result = tool.execute(tc.params)
                tc.tool_result = result
            except Exception as e:
                tc.tool_result = f"工具执行出错: {e}"

        # 记录到 ToolStateTracker（用于后续的上下文提示词）
        self._pctx.record_tool_call(ToolCallTrace(
            tool_name=tc.tool_name,
            success=True,
            summary=tc.tool_result[:120] if tc.tool_result else "",
        ))

        # 用工具结果 + LLM 合成最终答案
        # 例：工具返回 "北京：晴 22°C" → LLM 生成 "北京今天晴天，气温22°C..."
        synthesis_prompt = (f"{mem_prefix}\n\n你是一个全能的 AGI 智能助手。"
                           f"以下是你调用工具 '{tc.tool_name}' 获得的结果，"
                           f"请根据此结果回答用户问题。\n\n"
                           f"工具结果：{tc.tool_result}\n\n用户问题：{query}")
        answer = chat_llm(self, synthesis_prompt, [], on_event)
        step = ReActStep(type=StepType.ACTION, content=tc.tool_result,
                         tool=tc.tool_name, params=tc.params)
        return answer, [step]

    # ── 模式 3：RAG（知识库检索）──
    # 流程：改写查询 → 三路检索 → RRF 融合 → 重排 → LLM 合成
    def _run_rag(self, query: str, on_event: Callable = None) -> tuple:
        """
        RAG 知识库检索模式。

        RAG 引擎内部完整链路（见 src/domain/rag/engine.py）：
          1. LLM 改写查询（生成 3 个检索变体）
          2. 三路并行检索：Milvus（语义）+ ES（关键词）+ Neo4j（图谱）
          3. RRF 融合排序（倒数排名融合，去重 + 加权）
          4. 从 PG 加载匹配的文档块内容
          5. LLM 精排（Listwise 重新排序）
          6. LLM 合成最终答案（系统提示词 + 检索到的文档 + 用户问题）
        """
        # 取最近对话用于查询改写（上下文感知，消除指代歧义）
        history = recent_history_for_rag(self)
        answer, results = self._rag.query(query, history)
        emit(on_event, "rag_result", {"results": results})
        return answer, []

    # ── 模式 4：ReAct（多步推理）──
    # 三个 Phase：Planner 规划 → Executor 循环执行 → Generator 生成答案
    def _run_react(self, query: str, mem_prefix: str, history: List[Message],
                   opts: ChatOptions, on_event: Callable = None) -> tuple:
        """
        ReAct 多步推理模式：Planner → Executor 循环 → Generator

        适用场景：
          - 用户选了多个工具（"查天气 + 搜新闻 + 总结"）
          - 查询含 2+ 关键词（"北京天气怎么样？顺便查新闻"）
          - 需要多步推理的复杂问题

        三步走：
          Phase 1 — Planner（规划）：
            LLM 分析查询 + 可用工具列表 → 输出 JSON 执行计划
            如：[{step:1, tool:"get_weather", params:{city:"北京"}, reason:"查天气"},
                 {step:2, tool:"search_web",  params:{query:"北京新闻"}, reason:"查新闻"}]

          Phase 2 — Executor（循环执行）：
            按计划逐步执行每个工具，记录观察结果
            每步执行后：存快照到 PG（中断恢复）+ 写任务记忆

          Phase 3 — Generator（生成）：
            LLM 将所有步骤的观察结果合成为自然语言回答
        """
        # 过滤工具列表（用户可能只选了部分工具）
        tools = filter_tools(self, opts.selected_tools) if opts.selected_tools else self._tools.snapshot()
        task_id = f"task_{int(time.time() * 1000)}"

        # ═══ Phase 1: Planner 规划 ═══
        emit(on_event, "step", {"type": "planning", "content": "正在分析任务..."})

        # 构造工具描述列表（给 LLM 看，让它知道有哪些工具可用）
        tool_descriptions = "\n".join(
            f"- {name}: {t.description}" for name, t in tools.items()
        )

        # 规划提示词：让 LLM 输出 JSON 格式的执行步骤
        planner_prompt = (
            f"{mem_prefix}\n\n"
            "你是一个任务规划专家。根据用户查询和可用工具，制定执行计划。\n"
            f"可用工具：\n{tool_descriptions}\n\n"
            "输出 JSON 格式的执行步骤列表（最多5步）：\n"
            '[{"step": 1, "tool": "工具名", "params": {"参数名": "参数值"}, "reason": "原因"}]\n'
            "如果需要多步推理就输出多步，如果简单查询可以只有1步。"
        )

        # 调 LLM 生成执行计划（JSON 字符串）
        plan_raw = chat_llm(self, planner_prompt, history, on_event)

        # 解析 JSON（处理 LLM 可能输出的 markdown 代码块标记）
        steps = self._parse_plan(plan_raw)

        # 构建 TaskState 对象（追踪整个任务的执行状态）
        task = TaskState(task_id=task_id, query=query, status="running",
                         phase="executing", steps=[], current_step=0)
        for i, s in enumerate(steps):
            task.steps.append(TaskStep(
                id=i, name=s.get("reason", ""), tool_name=s.get("tool", ""),
                params=s.get("params", {}), status=TaskStepStatus.PENDING,
            ))
        self._runtime.set_task(task)
        self._pctx.reset_task_mem()  # 清空上轮任务记忆

        # ═══ Phase 2: Executor 循环执行 ═══
        react_steps: List[ReActStep] = []
        observations = []  # 收集所有步骤的观察结果

        for i, ts in enumerate(task.steps):
            task.current_step = i
            ts.status = TaskStepStatus.RUNNING

            # 找到对应工具
            tool = tools.get(ts.tool_name)
            if not tool or not tool.execute:
                ts.status = TaskStepStatus.FAILED
                ts.error = f"工具 {ts.tool_name} 不可用"
                continue

            # 通知前端：当前正在执行哪个步骤
            emit(on_event, "step", {
                "type": "action", "tool": ts.tool_name, "params": ts.params,
                "step": i + 1, "total": len(task.steps),
            })

            # 执行工具
            step = ReActStep(type=StepType.ACTION, tool=ts.tool_name, params=ts.params)
            try:
                result = tool.execute(ts.params)
                ts.result = result
                ts.status = TaskStepStatus.DONE
                step.content = result
                observations.append(f"[{ts.tool_name}] {result}")
            except Exception as e:
                ts.error = str(e)
                ts.status = TaskStepStatus.FAILED
                step.content = f"错误: {e}"
                observations.append(f"[{ts.tool_name}] 错误: {e}")

            react_steps.append(step)

            # ★ 写入任务记忆（供 PromptCtx 中的 TaskMemSource 读取）
            #   后续步骤的 LLM 调用可以看到前几步的结果
            self._pctx.push_task_mem(StepObservation(
                step_id=i, tool_name=ts.tool_name, result=ts.result,
                error=ts.error, success=ts.status == TaskStepStatus.DONE,
            ))

            # ★ 记录工具调用痕迹（供 ToolStateSource 读取）
            self._pctx.record_tool_call(ToolCallTrace(
                tool_name=ts.tool_name,
                success=ts.status == TaskStepStatus.DONE,
                summary=ts.result[:120] if ts.result else ts.error[:120],
            ))

            # ★ 保存快照到 PG（中断恢复：下次启动可以从此处继续）
            self._save_snapshot(task)

        # ═══ Phase 3: Generator 生成 ═══
        task.phase = "generating"
        emit(on_event, "step", {"type": "generating", "content": "正在总结结果..."})

        # 把所有步骤的观察结果拼接起来，让 LLM 生成最终答案
        obs_text = "\n".join(observations)
        generator_prompt = (
            f"{mem_prefix}\n\n"
            "你是一个结果总结专家。根据执行步骤的观察结果，生成最终回答。\n"
            f"用户问题：{query}\n"
            f"执行结果：\n{obs_text}\n\n"
            "请给出清晰完整的回答。"
        )
        answer = chat_llm(self, generator_prompt, [], on_event)

        # 标记任务完成
        task.status = "completed"
        task.phase = "done"
        task.result = answer
        react_steps.append(ReActStep(type=StepType.FINAL_ANSWER, content=answer))

        # 清理任务状态 + 保存最终快照
        self._runtime.set_task(None)
        self._save_snapshot(task)

        return answer, react_steps

    def _parse_plan(self, raw: str) -> list:
        """
        解析 LLM 输出的 JSON 执行计划。

        LLM 可能输出：
          - 纯 JSON：'[{"step": 1, "tool": "get_weather", ...}]'
          - Markdown 代码块：'```json\n[...]\n```'
          - 带多余空白

        解析失败时返回一个默认搜索步骤作为兜底。
        """
        try:
            # 去 markdown 代码块标记
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(raw)
        except Exception:
            # 兜底：至少尝试搜索
            return [{"step": 1, "tool": "search_web", "params": {"query": ""}, "reason": "默认搜索"}]

    def _save_snapshot(self, task: TaskState):
        """
        保存任务快照到 PostgreSQL + 内存。

        作用：ReAct 多步执行过程中，如果服务崩溃或用户中断，
        可以从快照恢复到上次执行到的位置继续。
        """
        try:
            data = json.dumps(task.__dict__, default=str).encode()
            self._repos.snap.save(task.task_id, data)
            self._runtime.append_snapshot(Snapshot(
                state=task, timestamp=time.strftime("%H:%M:%S"),
            ))
        except Exception as e:
            logger.warning(f"saveSnapshot failed: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    #  后处理（每轮对话后自动执行）
    # ═══════════════════════════════════════════════════════════════════════

    def _extract_and_save_prefs(self, msg: str):
        """
        从用户消息中提取偏好，并存到内存 + PG。

        两种提取方式（互补）：
          1. LLM 提取（主要）：prompt LLM 输出 JSON 格式偏好
             例："我喜欢简洁的回答" → {"回答风格": "简洁"}
          2. 正则提取（兜底）：匹配固定模式
             例："我叫张三" → {"姓名": "张三"}
             例："我在北京工作" → {"工作地": "北京"}

        为什么用 daemon 线程？偏好提取不影响响应速度，异步执行即可。
        """
        try:
            # LLM 提取（主要方式）
            prefs = self._llm.extract_preferences(msg)
            # 正则提取（兜底 + 补充）
            if not prefs:
                rule_prefs = self._mem.pref.extract_rule_based(msg)
                prefs.update(rule_prefs)
            # 双写：内存 + PG
            for k, v in prefs.items():
                self._mem.pref.save(k, v)
                self._repos.pref.save("default", k, v)
        except Exception as e:
            logger.warning(f"extractPrefs failed: {e}")

    def _store_long_term_memory(self, msg: str):
        """
        将用户消息存储为长期记忆。

        完整流程：
          1. 生成 Embedding 向量
          2. store_classified()：内存去重（cosine ≥ 0.95 视为重复）
          3. 持久化到 PG（新增 INSERT）
          4. 图索引：Neo4j 创建 Memory 节点 + FOLLOWS 边
          5. 合并检查：每 N 条新记忆触发 consolidate()（默认 N=10）
              - Phase 1: 重要性衰减（每日 ×0.995）
              - Phase 2: 去重合并（cosine ≥ 0.80 的相似记忆合并）
              - Phase 3: 过期淘汰（>30 天 + 重要性 < 0.3）
              - 同步 PG：DELETE 被淘汰的去重条目，UPDATE 内容/重要性变化条目
              - 同步 Neo4j：删除淘汰节点的图关系
        """
        try:
            # 生成 Embedding 向量
            emb = self._llm.embed(msg)

            # 分类存储（自动去重）
            item = self._mem.ltm.store_classified(
                content=msg, importance=0.5, embedding=emb,
                category="episodic",
            )

            if item:
                # 持久化到 PG
                self._repos.ltm.save_classified(
                    content=item.content, importance=item.importance,
                    embedding_json=json.dumps(item.embedding).encode(),
                    category=item.category, tags=item.tags,
                    slot_hint=item.slot_hint,
                )

                # Neo4j 图增强：建立 FOLLOWS 关系（时序链）
                if self._mem.graph_mem and self._mem.graph_mem.available():
                    prev = self._mem.ltm.snapshot()
                    prev_item = prev[-2] if len(prev) > 1 else None
                    self._mem.graph_mem.index_memory(item, prev_item)

            # 每 N 条新记忆触发合并检查（默认 10 条）
            if self._mem.ltm.need_consolidation():
                result = self._mem.ltm.consolidate()

                # ── 同步 PG：DELETE 被淘汰/去重的全部记忆 ──
                if result["deleted_ids"]:
                    self._repos.ltm.delete(result["deleted_ids"])
                    logger.info(f"长期记忆合并：已删除 {len(result['deleted_ids'])} 条")

                # ── 同步 PG：UPDATE 内容/重要性变化的记忆 ──
                for it in result["updated_items"]:
                    self._repos.ltm.update(
                        id=it.id, content=it.content, importance=it.importance,
                        embedding_json=json.dumps(it.embedding).encode(),
                    )

                # ── 同步 Neo4j ──
                if self._mem.graph_mem and self._mem.graph_mem.available():
                    # Phase 2 合并：保留节点吸收被删节点的关系，再删旧节点
                    for keep_id, remove_ids in result["merged_pairs"].items():
                        self._mem.graph_mem.merge_graph_nodes(keep_id, remove_ids)
                    # Phase 3 过期淘汰：直接删节点（DETACH DELETE 自动清理关系）
                    for eid in result["evicted_ids"]:
                        self._mem.graph_mem.delete_memory_node(eid)
        except Exception as e:
            logger.warning(f"storeLTM failed: {e}")
