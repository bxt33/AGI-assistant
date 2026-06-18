"""Context Builder：装配每轮推理需要的提示前缀和对话历史

=============================================================================
                    🔧 core_agent.py 的工具箱
=============================================================================

位置（调用链）：
  core_agent.Process()
    ├── build_context_prefix()  → 调 pctx 拼系统提示词前缀
    ├── build_history_messages() → 从 STM 取最近 N 轮对话
    └── chat_llm()              → 统一 LLM 调用入口（同步/流式）
        emit()                  → 流式事件回调（chat_llm 内部调用）

7 个函数各司其职：
  build_context_prefix(agent, query, mode) → Schema 驱动拼提示词前缀
  build_system_prompt(mem_prefix, base)    → 前缀 + 基础提示词拼接
  build_history_messages(agent, query)     → 取 STM 对话历史
  recent_history_for_rag(agent)            → 取最近 6 轮对话给 RAG 改写用
  filter_tools(agent, names)              → 按名字筛选工具
  emit(on_event, type, data)              → 流式事件通知前端
  chat_llm(agent, system, msgs, ...)      → 统一 LLM 调用（同步/流式）

设计理念：
  这些函数都只做"数据搬运"—不产生新数据，不修改状态（emit 除外）。
  比如 build_context_prefix 只是从 agent._pctx 读提示词，
  build_history_messages 只是从 agent._mem.stm 读历史，
  chat_llm 只是调 agent._llm。
  所有修改状态的操作都在 core_agent.py 里。
=============================================================================
"""

from typing import List, Optional, Callable

from src.application.chat.core_types import StreamEvent
from src.infrastructure.llm import Message
from src.domain.promptctx.source import Query as CtxQuery
from src.domain.tool import Tool
from src.domain.rag.rewriter import HistoryMessage


def build_context_prefix(agent, query: str, mode: str) -> str:
    """
    调 Schema 驱动的 ContextAssembler 输出系统提示词前缀。

    这是 Process() 步骤 3 的核心：
      输入：query="北京天气怎么样？", mode="tool"
      处理：
        ① 生成 query embedding（用于 LTM 语义召回）
        ② 获取当前 ReAct 任务 ID（如果有的话）
        ③ 调 agent._pctx.assemble(CtxQuery(...))
             → 选 Schema（按 mode）
             → 并发填充 Slot（6 个 Source 并发拉数据）
             → 预算裁剪 → RuntimeContext.render()
      输出（字符串）：
        【硬性约束】
        - 禁止删除根路径

        【用户画像】
        - 姓名: 张三
        - 回答风格: 简洁

    agent._pctx 为 None 时直接返回空字符串（启动期未初始化完成）。
    """
       if not hasattr(agent, '_pctx') or agent._pctx is None:

    emb = None
    if agent._llm:
        try:
            emb = agent._llm.embed(query)  # 为召回相关记忆生成向量
        except Exception:
            pass
    task = agent.current_task()
    task_id = task.task_id if task else ""
    # ★ 装配入口：所有 Schema、Source、Slot 的逻辑都在这里触发
    return agent._pctx.assemble(CtxQuery(
        text=query, embedding=emb, task_id=task_id, mode=mode,
    ))


def build_system_prompt(mem_prefix: str, base_prompt: str) -> str:
    """
    把 ContextAssembler 产出的前缀拼到基础提示词前面。

    如果前缀为空（pctx 未初始化），直接返回 base_prompt。

    例：
      base_prompt = "你是一个全能的 AGI 智能助手..."
      mem_prefix  = "【用户画像】\n- 姓名: 张三\n"
      输出        = "【用户画像】\n- 姓名: 张三\n\n你是一个全能的 AGI 智能助手..."
    """
    if not mem_prefix:
        return base_prompt
    return mem_prefix + "\n\n" + base_prompt


def build_history_messages(agent, query: str) -> List[Message]:
    """
    从 STM 滑动窗口取出最近 N 轮对话，拼成 LLM 可用的 Message 列表。

    STM 里存的是 (role, content) 对：
      [("user", "你好"), ("assistant", "你好！"), ("user", "帮我查天气"), ...]

    只取 role 为 "user" 或 "assistant" 的，过滤掉 system 消息。

    如果 STM 为空或最后一条不是当前 query，追加当前 query 到末尾。
    """
    msgs = []
    for m in agent._mem.stm.snapshot():
        if m.role in ("user", "assistant"):
            msgs.append(Message(role=m.role, content=m.content))
    if not msgs or msgs[-1].content != query:
        msgs.append(Message(role="user", content=query))
    return msgs


def recent_history_for_rag(agent) -> List[HistoryMessage]:
    """
    取最近 6 轮对话供 RAG 查询改写使用。

    为什么需要最近对话？
      用户在 RAG 模式下有时会说代词：
        "它怎么用？" ← 这里的"它"指什么？
      拿最近对话："我刚上传了 AGI-Saber 文档"
        → LLM 改写："AGI-Saber 怎么用？"（消除了指代歧义）

    只取最近 6 条（3 轮一问一答），足够消除指代，又不会太长。
    """
    snap = agent._mem.stm.snapshot()
    max_turns = 6
    start = max(0, len(snap) - max_turns)
    out = []
    for m in snap[start:]:
        if m.role in ("user", "assistant"):
            out.append(HistoryMessage(role=m.role, content=m.content))
    return out


def filter_tools(agent, names: List[str]) -> dict:
    """
    按名字列表筛选工具。

    用途：用户在前端勾选了工具（如 ["get_weather", "search_web"]），
    只把这些工具传给 ReAct Planner，而不是全部工具。
    """
    return agent._tools.filter(names)


def emit(on_event: Optional[Callable[[StreamEvent], None]], event_type: str, data):
    """
    流式事件通知前端。

    同步模式（/api/chat）：
      on_event = None → 什么都不做，直接跳过。

    流式模式（/api/chat/stream）：
      on_event 是一个回调，把事件塞进 handler.py 的 asyncio.Queue。
      最终通过 SSE（Server-Sent Events）推送给前端。

    事件类型：
      route      → 路由决策结果（chat/tool/rag/react）
      step       → ReAct 推理步骤（思考/动作/观察）
      tool_call  → 工具调用信息
      rag_result → RAG 检索结果
      token      → LLM 输出的文字片段（打字机效果）
      done       → 处理完成
    """
    if on_event:
        on_event(StreamEvent(type=event_type, data=data))


def chat_llm(agent, system: str, msgs: List[Message],
             on_event: Optional[Callable[[StreamEvent], None]] = None) -> str:
    """
    统一的 LLM 调用入口。

    根据是否有 on_event 回调分流：

    on_event = None（同步模式）：
      → agent._llm.chat(system, msgs)
      → 等 LLM 全部输出完，一次性返回完整文本。

    on_event != None（流式模式）：
      → agent._llm.chat_stream(system, msgs, on_token=...)
      → 每产出一个 token 就 emit("token", {"content": token})
      → 前端逐字显示（打字机效果）。

    这是整个项目唯一直接调 LLM 的地方（除了 RAG 查询改写和偏好提取中的专用调用）。
    """
    if on_event is None:
        # 同步模式：等全部返回
        return agent._llm.chat(system, msgs)
    # 流式模式：逐 token 推送
    return agent._llm.chat_stream(system, msgs, on_token=lambda t: emit(
        on_event, "token", {"content": t}))
