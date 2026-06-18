"""Query Rewriter：历史感知 + 多查询改写

=============================================================================
                    🔄 把 1 个问题变成 N 个检索变体
=============================================================================

位置（调用链）：
  engine.query()
    → LLMRewriter.rewrite(query, history)
    → 返回 ["原文", "变体1", "变体2", "变体3"]

设计目的：
  用户问"它怎么用？" → document store 里可能根本没有"它"这个词
  → LLM 改写为具体问题，消除指代歧义 + 从不同角度命中关键词

两种改写合并在一个 prompt 里完成：
  ① 指代消除：传入 history，让 LLM 知道"它" → "AGI-Saber"
  ② 变体生成：1 个问题 → N 个检索查询（不同关键词覆盖不同维度）

参数：
  num_queries = 3（默认，由 config.yaml: rag.rewrite.num_queries 控制）
  history = 最近 6 条对话（只用于指代消除，不用于变体）
=============================================================================
"""

from typing import List, Callable, Optional
from dataclasses import dataclass


@dataclass
class HistoryMessage:
    """标准化历史消息格式

    与 infrastructure/llm.py 的 Message 不同，这个是 RAG 专用的轻量结构，
    只包含 role 和 content 两个字段。
    """
    role: str      # "user" 或 "assistant"
    content: str   # 消息原文


class LLMRewriter:
    """用 LLM 将用户查询改写成多个不同角度的检索查询

    注入方式（见 core_agent._wire_rag_callbacks）：
      rewrite_llm = lambda s, u: self._llm.chat(s, [Message(role="user", content=u)])
      self._rag.set_rewriter(LLMRewriter(rewrite_llm, cfg.RAGRewriteNumQueries))
    """

    def __init__(self, llm_fn: Callable[[str, str], str], num_queries: int = 3):
        """
        参数：
          llm_fn      — LLM 调用函数：(system_prompt, user_msg) -> 回复文本
          num_queries — 改写变体数量（包含原文，最多返回 num_queries 条）
        """
        self._llm = llm_fn
        self._num_queries = num_queries

    def rewrite(self, query: str, history: Optional[List[HistoryMessage]] = None) -> List[str]:
        """
        将用户查询改写为多个检索变体。

        参数：
          query   — 用户原始问题（如 "它怎么配置？"）
          history — 最近对话历史，用于消除指代歧义

        返回：
          ["它怎么配置？", "AGI-Saber 配置方法", "AGI-Saber 配置文件教程"]

        工作流程：
          ① 把 history 拼成 "用户: xxx\n助手: xxx" 格式
          ② 拼 system prompt：要求输出 JSON 数组
          ③ 调 LLM → 解析 JSON → 确保原文在第一位 → 截到 num_queries 条
          ④ LLM 挂了 → 降级返回 [query]（只搜原文）
        """
        # ── ① 构建历史上下文文本 ──
        #     只取最近 6 条（3 轮问答），足够消除指代，又不浪费 tokens
        #     格式化为中文角色标签，方便 LLM 理解
        history_text = ""
        if history:
            parts = []
            for h in history[-6:]:
                role = "用户" if h.role == "user" else "助手"
                parts.append(f"{role}: {h.content}")
            history_text = "\n".join(parts)

        # ── ② 拼 LLM 提示词 ──
        #     要求输出 JSON 数组，不做 markdown 包装
        system = f"""你是一个查询改写专家。将用户的查询改写成 {self._num_queries} 个不同角度的检索查询。
每个查询应该从不同维度覆盖用户的需求，用于在知识库中检索相关信息。
直接输出 JSON 数组，不要有其他内容。"""

        # 如果有历史上下文，拼到用户消息里（告诉 LLM "它"指的是什么）
        user_msg = query
        if history_text:
            user_msg = f"对话历史：\n{history_text}\n\n当前查询：{query}"

        # ── ③ 调 LLM 并解析结果 ──
        try:
            import json
            raw = self._llm(system, user_msg)      # LLM 返回 '["变体1", "变体2", "变体3"]'
            raw = raw.strip().strip("```json").strip("```").strip()  # 去 markdown 标记
            queries = json.loads(raw)
            if isinstance(queries, list):
                result = [q for q in queries if isinstance(q, str)]
                # 确保原文始终在第一位（LLM 可能把原问题丢了）
                if query not in result:
                    result.insert(0, query)
                return result[:self._num_queries]   # 截断到配置的数量
        except Exception:
            pass

        # ── ④ LLM 挂了 → 降级：只返回原问题 ──
        return [query]
