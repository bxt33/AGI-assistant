"""Reranker：LLM Listwise 精排

=============================================================================
                    📊 从候选里挑最好的
=============================================================================

位置（调用链）：
  engine.query()
    → 召回 TopK×3（15 条）候选 chunk
    → LLMReranker.rerank(query, [chunk_0, chunk_1, ..., chunk_14])
    → 返回排序后的索引 [3, 0, 12, 7, 1, ...]
    → engine 取前 TopK（5 条）喂给 LLM 合成

为什么需要重排？
  向量相似（Milvus）≠ 真正相关
  BM25 高分（ES）   ≠ 真正相关

  例：用户问 "Python异步怎么用"
  Milvus 召回：chunk_3 "Python 协程简介" — 很相关！
  Milvus 召回：chunk_7 "Python 2.7 已停止维护" — 也有 "Python" 但不相关
  ES 召回：   chunk_12 "异步编程入门" — 很相关！

  RRF 融合后可能 chunk_7 的排名还不错（两路都有分数）
  → LLM 重排：读完 15 个候选的真实内容 → 把 chunk_7 排到最后

Listwise vs Pointwise vs Pairwise：
  - Pointwise：一篇一篇打分（慢，且没有对比）
  - Pairwise：两两比较（更慢）
  - Listwise：一次看完全部，直接给出排序 ← 这个项目用的
=============================================================================
"""

from typing import List, Callable


class LLMReranker:
    """用 LLM 对候选结果做 Listwise 精排

    注入方式（见 core_agent._wire_rag_callbacks）：
      rerank_llm = lambda s, u: self._llm.chat(s, [Message(role="user", content=u)])
      self._rag.set_reranker(LLMReranker(rerank_llm, cfg.RAGRerankPreviewLen))

    参数：
      llm_fn       — LLM 调用函数
      preview_len  — 每个 chunk 截取前 N 字符（默认 200），省 tokens
    """

    def __init__(self, llm_fn: Callable[[str, str], str], preview_len: int = 200):
        self._llm = llm_fn
        self._preview_len = preview_len

    def rerank(self, query: str, candidates: List[str]) -> List[int]:
        """
        对候选文档按相关性从高到低排序。

        参数：
          query      — 用户原始问题（"Python异步怎么用"）
          candidates — 候选 chunk 原文列表，15 条，每条约 200 字

        返回：
          排序后的索引列表，如 [3, 0, 12, 7, 1, ...]
          索引 3 的 chunk 最相关，索引 1 的最不相关

        工作流程：
          ① 每个候选截取前 preview_len 字符做预览（省 tokens）
          ② 拼 system prompt 要求输出 JSON 数组
          ③ LLM 一次性看完 15 个预览 → 输出 [3, 0, 12, ...]
          ④ 校验：只保留有效索引 → 补上缺失的（防御 LLM 漏输出）
          ⑤ LLM 挂了 → 降级返回原始顺序 [0, 1, 2, ...]
        """
        # ── 不足 2 条，不需要重排 ──
        if len(candidates) <= 1:
            return list(range(len(candidates)))

        # ── ① 截取预览 ──
        #     不传全文，只传前 200 字符——足够判断相关性，节省大量 tokens
        previews = []
        for i, c in enumerate(candidates):
            preview = c[:self._preview_len]
            previews.append(f"[{i}] {preview}")    # "[0] Python是一种解释型语言..."

        # ── ② 拼 LLM 提示词 ──
        #     Listwise：LLM 一次看完所有候选，给出全局排序
        system = """你是一个检索结果排序专家。根据用户查询，对候选文档按相关性从高到低排序。
直接输出 JSON 数组，包含排序后的索引，例如 [2, 0, 1, 3]。
只输出 JSON 数组，不要有其他内容。"""

        user_msg = f"查询：{query}\n\n候选文档：\n" + "\n\n".join(previews)

        # ── ③ 调 LLM 并解析 ──
        try:
            import json
            raw = self._llm(system, user_msg)
            raw = raw.strip().strip("```json").strip("```").strip()
            order = json.loads(raw)     # [3, 0, 12, 7, 1, 2, 5, ...]

            if isinstance(order, list) and all(isinstance(x, int) for x in order):
                # ④ 只保留有效索引（0 <= i < len(candidates)）
                valid = [i for i in order if 0 <= i < len(candidates)]
                # 补上 LLM 可能漏掉的索引（排在最后）
                for i in range(len(candidates)):
                    if i not in valid:
                        valid.append(i)
                return valid
        except Exception:
            pass

        # ── ⑤ LLM 挂了 → 降级：保持原始顺序 ──
        return list(range(len(candidates)))
