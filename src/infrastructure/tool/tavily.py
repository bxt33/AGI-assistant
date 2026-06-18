"""Tavily Search API 客户端"""

import json
from typing import Optional

import httpx


def tavily_search(query: str, api_key: str, api_url: str = "") -> str:
    """调用 Tavily Search API，返回格式化的搜索结果摘要"""
    if not api_url:
        api_url = "https://api.tavily.com/search"

    body = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": 5,
    }

    try:
        resp = httpx.post(api_url, json=body, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()

        answer = data.get("answer", "")
        results = data.get("results", [])

        # 优先返回 Tavily 合成的 answer
        if answer:
            parts = [answer]
            if results:
                parts.append("\n**来源：**")
                for r in results[:3]:
                    parts.append(f"- [{r.get('title', '')}]({r.get('url', '')})")
            return "\n".join(parts)

        # 无 answer 时拼接 top 结果
        if not results:
            return "Tavily 返回空结果"

        parts = []
        for r in results[:3]:
            parts.append(f"**{r.get('title', '')}**\n{r.get('content', '')}\n{r.get('url', '')}\n")
        return "\n".join(parts)

    except Exception as e:
        return f"Tavily 搜索失败: {e}"
