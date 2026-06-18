"""HTTP API 路由处理（FastAPI）

=============================================================================
后端请求入口：所有前端请求通过 HTTP API 进入，最终都调用 agent.Process()
=============================================================================

架构位置：
  main.py → create_app(agent) → 注册以下 10 个端点 → 启动 uvicorn

  浏览器 ──HTTP──► FastAPI ──► agent.Process(message, opts) ──► UnifiedAgent 核心

端点总览：
  POST /api/chat          — 同步聊天（一问一答，等待完整结果）
  POST /api/chat/stream   — 流式聊天（SSE 实时推送 token，打字机效果）
  POST /api/chat/cancel   — 取消正在执行的任务
  POST /api/upload        — 上传文档（RAG 知识库）
  POST /api/docs/delete   — 删除文档
  GET  /api/memory        — 查看记忆状态（调试用）
  GET  /api/tools         — 列出可用工具
  POST /api/tools/mcp     — 注册外部 MCP 工具
  GET  /api/snapshots     — 查看 ReAct 任务快照
  GET  /api/status        — 系统健康状态
"""

import json
import asyncio
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.application.chat.core_types import ChatOptions, StreamEvent
from src.domain.tool import Param as ToolParam
from src.infrastructure.tool.mcp import new_mcp_tool


def create_app(agent, cfg) -> FastAPI:
    """
    创建 FastAPI 应用，绑定 UnifiedAgent 到所有 HTTP 路由。

    参数：
      agent — UnifiedAgent 实例（核心大脑，在 main.py 中创建）
      cfg   — APIConfig 配置对象

    返回：
      FastAPI 应用实例，由 main.py 中的 uvicorn.run() 启动

    架构关键点：
      所有 HTTP 端点内部都只做三件事：
        1. 解析请求体 → 构造 ChatOptions
        2. 调用 agent.Process() 或 agent 的其他方法
        3. 返回 JSON 给前端
      真正的业务逻辑（路由、推理、记忆、工具）全在 agent 内部。
      这里只是"传话的"——把 HTTP 请求翻译成 agent 方法调用。
    """
    app = FastAPI(title="AGI Saber", description="AGI智能助手 API")

    # CORS 中间件：允许前端跨域访问（开发用，生产需收紧）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ══════════════════════════════════════════════════════════════
    # POST /api/chat — 同步聊天（核心入口）
    #
    # 请求体：
    #   { "message": "你好", "use_rag": false, "selected_tools": [], "explicit": false }
    #
    #   - message:       用户输入的消息文本
    #   - use_rag:       是否开启知识库检索（前端"知识库"开关）
    #   - selected_tools: 用户勾选的工具列表（如 ["get_weather", "search_web"]）
    #   - explicit:      是否显式指定模式（为 False 时走自动路由）
    #
    # 处理流程（内部）：
    #   HTTP 请求 → 解析 ChatOptions → agent.Process(message, opts) → 返回 Response
    #
    #   agent.Process() 内部做的事（详见 core_agent.py）：
    #     ① 存短期记忆（STM）
    #     ② 路由判断：chat / tool / rag / react
    #     ③ 拼系统提示词（Schema 驱动：约束+画像+回忆+工具状态）
    #     ④ 按模式执行（调 LLM / 查知识库 / 执行工具 / ReAct 推理）
    #     ⑤ 后处理：存记忆、写 PG、异步提取偏好
    #     ⑥ 返回 Response
    #
    # 响应：
    #   {
    #     "query": "你好",                    // 原始问题
    #     "answer": "你好！有什么可以帮你？",   // AI 回答
    #     "mode": "chat",                     // 走了哪种模式
    #     "steps": [...],                     // ReAct 模式下的思考步骤
    #     "tool_call": {...},                 // 工具调用详情
    #     "search_results": [...],            // RAG 检索结果
    #     "short_term_count": 2,              // 短期记忆条数
    #     "long_term_count": 15,              // 长期记忆条数
    #     "preferences": {...},               // 用户偏好
    #     "interrupted": false                // 是否被中断
    #   }
    # ══════════════════════════════════════════════════════════════
    @app.post("/api/chat")
    async def chat(request: Request):
        # 解析请求体
        body = await request.json()
        msg = body.get("message", "")

        # 构造 ChatOptions（前端开关 → 后端选项）
        #   use_rag:        对应前端"知识库"开关
        #   selected_tools: 对应前端工具多选框
        #   explicit:       True=用户手动指定了模式，跳过自动路由
        opts = ChatOptions(
            use_rag=body.get("use_rag", False),
            selected_tools=body.get("selected_tools"),
            explicit=body.get("explicit", False),
        )

        # ★ 核心调用：所有逻辑都在这一行之后
        # agent.Process() 是后端最重要的方法，详见 core_agent.py
        resp = agent.Process(msg, opts)

        # 把 Response 对象序列化为 JSON 返回前端
        return {
            "query": resp.query, "answer": resp.answer, "mode": resp.mode,
            "steps": [s.__dict__ for s in resp.steps],
            "tool_call": resp.tool_call.__dict__ if resp.tool_call else None,
            "search_results": resp.search_results,
            "short_term_count": resp.short_term_count,
            "long_term_count": resp.long_term_count,
            "preferences": resp.preferences,
            "interrupted": resp.interrupted,
        }

    # ══════════════════════════════════════════════════════════════
    # POST /api/chat/stream — SSE 流式聊天
    #
    # 和 /api/chat 的区别：
    #   同步版 /api/chat      → 等 AI 全部回答完，一次性返回 JSON
    #   流式版 /api/chat/stream → 实时推送 token，前端逐字显示（打字机效果）
    #
    # 实现方式：
    #   用 asyncio.Queue 做桥接 ——
    #   agent.ProcessStream() 在后台线程中处理，
    #   每产生一个 token 就通过 on_event 回调写入 queue，
    #   异步生成器 event_generator() 从 queue 读出来发给前端。
    #
    # SSE 事件类型（前端 switch-case 处理）：
    #   route      → 路由决策（走了哪种模式）
    #   step       → ReAct 推理步骤（思考/动作/观察）
    #   tool_call  → 工具调用信息
    #   rag_result → RAG 检索结果
    #   memory     → 记忆提取信息
    #   token      → LLM 输出的文字片段（前端逐字追加）
    #   done       → 处理完成
    # ══════════════════════════════════════════════════════════════
    @app.post("/api/chat/stream")
    async def chat_stream(request: Request):
        body = await request.json()
        msg = body.get("message", "")
        opts = ChatOptions(
            use_rag=body.get("use_rag", False),
            selected_tools=body.get("selected_tools"),
            explicit=body.get("explicit", False),
        )

        async def event_generator():
            """SSE 事件生成器：从 queue 中逐个取出事件，yield 给前端"""
            queue = asyncio.Queue()

            # on_event 回调：agent 每产生一个事件就放入队列
            # 这个回调在后台线程中被调用（run_in_executor）
            def on_event(evt: StreamEvent):
                data = evt.data
                if hasattr(data, '__dict__'):
                    data = data.__dict__
                # SSE 格式：event: <类型>\ndata: <JSON>\n\n
                queue.put_nowait(f"event: {evt.type}\ndata: {json.dumps(data, default=str)}\n\n")

            # 在后台线程中运行 agent.ProcessStream（同步方法，会阻塞）
            # 这样不会阻塞 FastAPI 的事件循环
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, lambda: agent.ProcessStream(msg, opts, on_event))

            # 从队列中读取事件并 yield
            # wait_for 设置 120 秒超时，防止 LLM 卡死导致连接永不关闭
            while True:
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=120)
                    yield chunk
                    if '"event: done"' in chunk or 'event: done' in chunk:
                        break
                except asyncio.TimeoutError:
                    break

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # ══════════════════════════════════════════════════════════════
    # POST /api/chat/cancel — 取消正在执行的任务
    #
    # 当前端用户点"停止"按钮时调用。
    # 实现方式：设置 agent 内部的取消信号，正在执行的 LLM 调用/工具执行检查后会中断。
    # ══════════════════════════════════════════════════════════════
    @app.post("/api/chat/cancel")
    async def chat_cancel():
        agent.Cancel()
        return {"ok": True, "message": "已发送取消信号"}

    # ══════════════════════════════════════════════════════════════
    # POST /api/upload — 上传文档到知识库（RAG）
    #
    # 前端把文件内容读成文本，POST 到这里。
    # agent.RAG().ingest(content) 做的事：
    #   1. 文本分块（滑动窗口，chunk_size=200, overlap=50）
    #   2. 生成 Embedding 向量
    #   3. 写入 PostgreSQL（块内容）+ Milvus（向量）+ ES（全文索引）
    #   返回 doc_hash（用于后续删除）
    # ══════════════════════════════════════════════════════════════
    @app.post("/api/upload")
    async def upload(request: Request):
        body = await request.json()
        content = body.get("content", "")
        count, doc_hash = agent.RAG().ingest(content)
        return {"chunk_count": count, "doc_hash": doc_hash,
                "chunks": [{"id": c.id, "content": c.content[:100]} for c in agent.RAG().chunks()]}

    # POST /api/docs/delete — 从知识库中删除指定文档
    @app.post("/api/docs/delete")
    async def docs_delete(request: Request):
        body = await request.json()
        doc_hash = body.get("doc_hash", "")
        if not doc_hash:
            raise HTTPException(status_code=400, detail="doc_hash is required")
        agent.RAG().delete(doc_hash)  # 从 PG+Milvus+ES 三端同时删除
        return {"ok": True, "doc_hash": doc_hash}

    # ══════════════════════════════════════════════════════════════
    # GET /api/memory — 查看记忆状态（调试用）
    #
    # 返回三层记忆的当前快照：
    #   short_term:  [{role="user", content="你好", ...}, ...]
    #   long_term:   [{id, content, importance, category, tags, score}, ...]
    #   preference:  {name: "张三", 喜好: "编程", ...}
    # ══════════════════════════════════════════════════════════════
    @app.get("/api/memory")
    async def memory():
        return {
            "short_term": [m.__dict__ for m in agent.ShortTerm().snapshot()],
            "long_term": [{
                "id": it.id, "content": it.content,
                "importance": it.importance, "category": it.category,
                "tags": it.tags, "score": it.score,
            } for it in agent.LongTerm().snapshot()],
            "preference": agent.Preferences().snapshot(),
        }

    # ══════════════════════════════════════════════════════════════
    # GET /api/tools — 列出所有可用工具
    #
    # 返回给前端渲染工具选择框。
    # 工具来源：内置工具（time/weather/search）+ exec_command + rag_search + MCP 注册工具
    # ══════════════════════════════════════════════════════════════
    @app.get("/api/tools")
    async def tools_list():
        tools = agent.Tools()
        return [{
            "name": t.name, "description": t.description,
            "is_mcp": t.is_mcp,
            "params": [p.__dict__ for p in t.parameters],
        } for t in tools.values()]

    # POST /api/tools/mcp — 动态注册外部 MCP 工具
    # MCP（Model Context Protocol）：通过 HTTP 调用外部服务作为工具
    @app.post("/api/tools/mcp")
    async def register_mcp_tool(request: Request):
        body = await request.json()
        name = body.get("name", "")
        endpoint = body.get("endpoint", "")
        if not name or not endpoint:
            raise HTTPException(status_code=400, detail="name and endpoint are required")
        params = [ToolParam(**p) for p in body.get("params", [])]
        t = new_mcp_tool(name, body.get("description", ""), endpoint, params)
        agent.RegisterTool(t)  # 运行时动态注册，不需要重启
        return {"ok": True, "name": name}

    # GET /api/snapshots — 查看 ReAct 任务快照
    # ReAct 模式每步执行后会保存快照到 PG，用于中断恢复
    @app.get("/api/snapshots")
    async def snapshots():
        snaps = agent.Snapshots()
        return [{
            "index": i, "timestamp": s.timestamp,
            "steps": len(s.state.steps),
        } for i, s in enumerate(snaps)]

    # GET /api/status — 系统健康状态
    # 返回 RAG 状态、记忆计数、当前 LLM 模型、基础设施连接状态
    @app.get("/api/status")
    async def status():
        return agent.Status()

    # ── 挂载前端静态文件（frontend/index.html）──
    # 访问 http://localhost:8090 直接看到前端页面
    try:
        app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
    except Exception:
        pass

    return app
