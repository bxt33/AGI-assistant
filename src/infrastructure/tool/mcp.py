"""MCP 工具：调用外部 HTTP 端点"""

import json
from typing import Dict, Any, List

import httpx

from src.domain.tool import Tool, Param


def new_mcp_tool(name: str, description: str, endpoint: str,
                 params: List[Param]) -> Tool:
    """创建一个调用外部 HTTP 端点的 MCP 兼容工具"""

    def _exec(p: Dict[str, Any]) -> str:
        try:
            resp = httpx.post(endpoint, json=p, timeout=30.0)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            return f"MCP 请求失败 [{endpoint}]: {e}"

    return Tool(
        name=name,
        description=description,
        parameters=params,
        is_mcp=True,
        execute=_exec,
    )
