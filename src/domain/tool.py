"""工具抽象：定义 Agent 可调用的工具与基于规则的工具选择逻辑"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field


@dataclass
class Param:
    name: str
    type: str = "string"
    description: str = ""
    required: bool = False


@dataclass
class Tool:
    name: str
    description: str
    parameters: List[Param] = field(default_factory=list)
    is_mcp: bool = False
    execute: Optional[Callable[[Dict[str, Any]], str]] = None


@dataclass
class CallResult:
    tool_name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    tool_result: str = ""


def decide(query: str, tools: Dict[str, Tool]) -> Optional[CallResult]:
    """基于规则推断应调用的工具及参数"""
    q = query.lower()

    # 时间
    if "几点" in q or "时间" in q:
        if "get_time" in tools:
            params: Dict[str, Any] = {}
            if "东京" in q:
                params["timezone"] = "Asia/Tokyo"
            return CallResult(tool_name="get_time", params=params)

    # 天气
    if "天气" in q:
        if "get_weather" in tools:
            city = "北京"
            for c in ["东京", "北京", "上海", "纽约", "伦敦", "广州", "深圳"]:
                if c in q:
                    city = c
                    break
            return CallResult(tool_name="get_weather", params={"city": city})

    # 搜索
    if any(kw in q for kw in ["查", "搜索", "是什么"]):
        if "search_web" in tools:
            return CallResult(tool_name="search_web", params={"query": query})

    # exec_command 兜底
    if "exec_command" in tools:
        return CallResult(tool_name="exec_command", params={"command": query})

    # 取第一个工具兜底
    for name, t in tools.items():
        param_name = "query"
        for p in t.parameters:
            if p.required:
                param_name = p.name
                break
        return CallResult(tool_name=name, params={param_name: query})

    return None
