"""内置工具实现：time / weather / search_web"""

import time as _time
from typing import Dict, Any

from src.domain.tool import Tool, Param


def get_time() -> Tool:
    return Tool(
        name="get_time",
        description="获取当前时间",
        parameters=[
            Param(name="timezone", type="string", description="时区（如 Asia/Tokyo）", required=False),
        ],
        execute=lambda p: _time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def get_weather() -> Tool:
    db = {
        "北京": "晴天 22°C",
        "东京": "多云 18°C 湿度65%",
        "上海": "小雨 20°C",
        "纽约": "晴天 15°C",
        "伦敦": "阴天 12°C",
        "广州": "晴天 28°C",
        "深圳": "晴天 26°C",
    }

    def _exec(params: Dict[str, Any]) -> str:
        city = params.get("city", "北京")
        if isinstance(city, str) and city in db:
            return f"{city}：{db[city]}"
        return f"{city}：晴天 20°C（模拟）"

    return Tool(
        name="get_weather",
        description="获取城市天气信息",
        parameters=[
            Param(name="city", type="string", description="城市名称", required=True),
        ],
        execute=_exec,
    )


def search_web() -> Tool:
    db = {
        "AI应用工程师": "AI 应用工程师是将 AI 技术落地到业务的工程师，需具备 ML 基础、API 开发、Prompt 工程等能力。",
        "Go语言": "Go 是 Google 开发的开源编程语言，适用于高并发服务端应用。Docker 即用 Go 开发。",
    }

    def _exec(params: Dict[str, Any]) -> str:
        q = params.get("query", "")
        if isinstance(q, str):
            for k, v in db.items():
                if k in q:
                    return v
        return f"关于「{q}」的搜索结果（模拟）"

    return Tool(
        name="search_web",
        description="搜索互联网获取最新信息",
        parameters=[
            Param(name="query", type="string", description="搜索关键词", required=True),
        ],
        execute=_exec,
    )


def default_tools() -> Dict[str, Tool]:
    tools = [get_time(), get_weather(), search_web()]
    return {t.name: t for t in tools}
