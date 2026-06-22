"""工具抽象：定义 Agent 可调用的工具与基于规则的工具选择逻辑。

=============================================================================
                    🔧 工具系统：Agent 的手和脚
=============================================================================

Agent 不是只能聊天的——它能调用工具做实事。"工具"就是 LLM 之外的执行能力。

═══════════════════════════════════════════════════════════════════
                    两层工具选择
═══════════════════════════════════════════════════════════════════

  方式 1（主路径）：LLM function calling
    core_agent._run_tool() 把工具定义发给大模型
    → 大模型理解用户意图，返回 {tool: "get_weather", params: {city: "杭州"}}
    → 准确、灵活，任何城市都能识别

  方式 2（降级路径）：规则匹配 decide()  ← 就在这个文件里
    LLM 不可用或 function calling 失败时，用关键词匹配推断用户想调什么工具
    → 零成本（没有 LLM 调用），但只能覆盖预设的关键词模式

═══════════════════════════════════════════════════════════════════
                    数据结构
═══════════════════════════════════════════════════════════════════

  Tool      — 工具定义（名称、描述、参数列表、是否 MCP、执行函数）
  Param     — 工具参数（名称、类型、描述、是否必填）
  CallResult— 工具调用结果（工具名、参数、返回值），在 Agent 和 Sandbox 之间传递

═══════════════════════════════════════════════════════════════════
                    工具注册
═══════════════════════════════════════════════════════════════════

  工具在 UnifiedAgent._init_tools() 中注册（见 core_agent.py），包括：
    get_time      — 获取当前时间（支持指定时区）
    get_weather   — 查询指定城市的天气
    search_web    — 网络搜索
    exec_command  — 执行 Shell 命令（危险操作，受沙箱和安全策略限制）
    rag_search    — 从私人知识库检索文档（走 RAG 三路混合检索）

  每个工具都有：
    • description — 告诉 LLM 这个工具干什么
    • parameters  — 告诉 LLM 需要传什么参数（name, type, required）
    • execute     — 实际执行函数（Python callable），不是在 LLM 那边执行

=============================================================================
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════
#  数据结构
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Param:
    """
    工具参数定义。

    用于生成 LLM function calling 的 JSON Schema，告诉大模型这个参数叫什么、
    什么类型、必填还是可选。

    Attributes:
        name:        参数名，如 "city", "timezone", "query"
        type:        参数类型，如 "string", "number", "boolean"
        description: 参数描述，如 "城市名称，如北京" — LLM 据此理解参数含义
        required:    是否必填。True = 工具调用必须提供此参数
    """
    name: str
    type: str = "string"
    description: str = ""
    required: bool = False


@dataclass
class Tool:
    """
    工具定义。

    每个 Tool 对象代表一个 Agent 可调用的工具。在工具注册表（dict）中以 name 为 key。

    两种来源：
      ① 内置工具（get_time, get_weather, search_web, exec_command）
         — execute 是 Python 函数，在 Sandbox 中执行
      ② MCP 工具（外部服务，通过 MCP 协议接入）
         — is_mcp=True，execute 可能为 None（由 MCP 客户端远程调用）

    Attributes:
        name:        工具名称（唯一标识），如 "get_weather"
        description: 工具描述，LLM 据此判断何时调用——写得越清楚越好
        parameters:  参数列表（List[Param]），告诉 LLM 调用时要传什么参数
        is_mcp:      是否 MCP（外部）工具
        execute:     实际执行函数。签名：fn(params) -> str
                     内置工具实现如 def get_weather(params): ...
    """
    name: str
    description: str
    parameters: List[Param] = field(default_factory=list)
    is_mcp: bool = False
    execute: Optional[Callable[[Dict[str, Any]], str]] = None


@dataclass
class CallResult:
    """
    工具调用结果——在 Agent 决策层 和 Sandbox 执行层 之间传递。

    生命周期：
      ① Agent 决定调用什么工具 → 创建 CallResult(tool_name, params)
      ② Sandbox 根据 tool_name 找到对应的 execute 函数
      ③ 执行后把结果写入 tool_result
      ④ Agent 把 tool_result 作为工具输出展示给用户或传给 LLM 下一步思考

    Attributes:
        tool_name:   被调用的工具名
        params:      调用参数，如 {"city": "北京"}
        tool_result: 工具执行结果文本，如 "北京 晴 22°C"
    """
    tool_name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    tool_result: str = ""


# ═══════════════════════════════════════════════════════════════════════════
#  规则匹配：降级路径（LLM function calling 不可用时）
# ═══════════════════════════════════════════════════════════════════════════

def decide(query: str, tools: Dict[str, Tool]) -> Optional[CallResult]:
    """
    基于规则推断应调用的工具及参数。

    ⚠️ 这是降级路径——只在 LLM function calling 不可用时使用。
    主路径是 LLM 理解用户意图后返回 function_call。

    ── 规则匹配逻辑 ──
    按优先级依次检查 query 中的关键词：

    ① "几点" / "时间" → get_time
       如果含 "东京" → 自动设置 timezone="Asia/Tokyo"
       否则 → 默认时区

    ② "天气" → get_weather
       从预定义的 7 个城市列表中匹配，默认 "北京"
       为什么只有 7 个城市？规则匹配是关键词检测，不是 NLP。
       做不到从 "杭州" 这样的任意城市名中提取——真正灵活的提取靠 LLM。

    ③ "查" / "搜索" / "是什么" → search_web
       把整个 query 作为搜索关键词

    ④ 兜底 exec_command → 把整个 query 作为命令执行

    ⑤ 再兜底 → 取工具注册表中第一个工具的 query 参数

    ── 调用时机 ──
    core_agent._run_tool() 中：
      result = self._llm.function_call(query, tools)   ← LLM 主路径
      if result is None:
          result = decide(query, tools)                 ← 规则降级

    ── 为什么不是每个工具都支持规则匹配？ ──
    规则匹配本质是 if/else 堆砌。每增加一个工具的规则支持，就要增加
    关键词判断逻辑。维护成本高，覆盖率有限。正确的方向是让 LLM
    function calling 更可靠，而不是让规则更复杂。

    Args:
        query: 用户原始查询文本
        tools: 可用工具注册表 {name: Tool}

    Returns:
        CallResult 如果匹配到工具和参数，None 如果完全没有匹配。
    """
    q = query.lower()

    # ── ① 时间 ──
    #     关键词 "几点" 或 "时间"
    #     如果提到东京 → 自动设置 Asia/Tokyo 时区（其他时区同理可扩展）
    if "几点" in q or "时间" in q:
        if "get_time" in tools:
            params: Dict[str, Any] = {}
            if "东京" in q:
                params["timezone"] = "Asia/Tokyo"
            return CallResult(tool_name="get_time", params=params)

    # ── ② 天气 ──
    #     关键词 "天气"
    #     城市从预定义列表中匹配，默认北京
    #     局限性：不支持列表外的城市（如"杭州"），这需要 LLM 理解
    if "天气" in q:
        if "get_weather" in tools:
            city = "北京"
            for c in ["东京", "北京", "上海", "纽约", "伦敦", "广州", "深圳"]:
                if c in q:
                    city = c
                    break
            return CallResult(tool_name="get_weather", params={"city": city})

    # ── ③ 搜索 ──
    #     关键词 "查" / "搜索" / "是什么" → 把整个 query 当搜索词
    if any(kw in q for kw in ["查", "搜索", "是什么"]):
        if "search_web" in tools:
            return CallResult(tool_name="search_web", params={"query": query})

    # ── ④ 命令执行兜底 ──
    #     如果前面都不匹配，但有 exec_command，就用整个 query 作为命令
    #     这是最激进的兜底——可能会把"今天天气怎么样"当成命令执行
    if "exec_command" in tools:
        return CallResult(tool_name="exec_command", params={"command": query})

    # ── ⑤ 终极兜底 ——
    #     取工具注册表中第一个工具的 query 参数
    #     这个几乎不会走到，但作为安全网保留
    for name, t in tools.items():
        param_name = "query"
        for p in t.parameters:
            if p.required:
                param_name = p.name
                break
        return CallResult(tool_name=name, params={param_name: query})

    return None
