"""混合检索：RRF（Reciprocal Rank Fusion）融合排序

=============================================================================
                    🔀 把多路检索结果合并成一张最终排名
=============================================================================

位置（调用链）：
  engine.query()
    → Milvus 返回: [{pg_id:7, score:0.92}, {pg_id:3, score:0.87}, ...]
    → ES 返回:     [{pg_id:7, score:8.5}, {pg_id:12, score:7.2}, ...]
    → KG 返回:     [{pg_id:7, score:1.0}, {pg_id:15, score:0.8}, ...]
    → rrf_fusion(milvus_hits, es_hits, kg_hits)
    → 返回 [7, 3, 12, 5, ...]  （按融合分降序排列的 pg_id 列表）

RRF 核心思想：
  不看原始分数（不同搜索引擎打分尺度不同，无法直接比较），
  只看排名 — 排名越靠前，贡献越大。

  公式：score(doc) = Σ 1/(k + rank(doc))
  其中 k=60（经典平滑常数），rank 从 0 开始

  为什么 k=60？
    - k 太大 → 所有排名分数趋同，区分度低
    - k 太小 → 排名 0 的分数太高（1/1=1.0），排名 10 的太低（1/11=0.09），差异过大
    - k=60 是学术论文验证的平衡值

  效果：两个引擎都排在前面的文档分数最高，只有一个引擎排前面的会降权。

示意：
  Milvus 排名: [A(0), C(1), B(2)]
  ES 排名:     [B(0), A(1), D(2)]

  A 的 RRF = 1/(60+0)×0.7 + 1/(60+1)×1.0 = 0.0117 + 0.0164 = 0.0281
  B 的 RRF = 1/(60+2)×0.7 + 1/(60+0)×1.0 = 0.0113 + 0.0167 = 0.0280
  C 的 RRF = 1/(60+1)×0.7 + 0             = 0.0115               （只有一路）
  D 的 RRF = 0             + 1/(60+2)×1.0 = 0.0161               （只有一路）

  最终排序：A > B > D > C
  ↑ A 和 B 两路都高 → 最可靠，D 和 C 只有单路 → 降权
=============================================================================
"""

from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    """标准化搜索结果（当前未直接使用，留给未来扩展）"""
    pg_id: int = 0         # 关联 PG rag_chunks 表的主键
    score: float = 0.0     # 搜索引擎给的原始分数
    content: str = ""      # chunk 原文（可选）
    source: str = ""       # 来源："milvus" | "es" | "neo4j"


def rrf_fusion(
    milvus_hits: List[Dict],
    es_hits: List[Dict],
    kg_hits: List[Dict],
    k: int = 60,
    semantic_weight: float = 0.7,
    kg_weight: float = 0.3,
) -> List[int]:
    """
    RRF (Reciprocal Rank Fusion) 融合多路检索结果。

    参数：
      milvus_hits     — Milvus 向量搜索结果 [{"pg_id": 7, "score": 0.92}, ...]
      es_hits         — ES 关键词搜索结果  [{"pg_id": 7, "score": 8.5}, ...]
      kg_hits         — Neo4j 图遍历结果   [{"pg_id": 7, "score": 1.0}, ...]
      k               — RRF 平滑常数（默认 60）
      semantic_weight — Milvus 语义检索权重（默认 0.7，可在配置中调）
      kg_weight       — Neo4j 图谱检索权重（默认 0.3，半权——概念关联不如直接匹配可靠）

    返回：
      按融合分降序排列的 pg_id 列表，如 [7, 3, 12, 5, 1, ...]

    权重设计：
      Milvus 语义 = 0.7  （主力，语义相似匹配最可靠）
      ES 关键词   = 1.0  （基础权重，BM25 关键词匹配也很可靠，没写参数里）
      Neo4j 图谱  = 0.3  （辅助，概念关联可能过于发散）
    """
    scores: Dict[int, float] = {}

    # ── Milvus 语义向量结果 ──
    #     weight = 0.7（语义相似匹配最可靠）
    for rank, hit in enumerate(milvus_hits):
        pg_id = hit.get("pg_id", hit.get("id", 0))
        if pg_id:
            # RRF 公式：1 / (k + rank + 1)，rank 从 0 开始
            # 例：排名第 1  → 1/(60+0+1) = 1/61 = 0.0164
            #     排名第 10 → 1/(60+9+1) = 1/70 = 0.0143
            rrf_score = 1.0 / (k + rank + 1) * semantic_weight
            scores[pg_id] = scores.get(pg_id, 0) + rrf_score

    # ── ES BM25 关键词结果 ──
    #     weight = 1.0（基础权重，和语义检索同等重要）
    for rank, hit in enumerate(es_hits):
        pg_id = hit.get("pg_id", 0)
        if pg_id:
            rrf_score = 1.0 / (k + rank + 1)   # 权重 = 1.0
            scores[pg_id] = scores.get(pg_id, 0) + rrf_score

    # ── Neo4j 知识图谱结果 ──
    #     weight = 0.3（辅助，概念关系可能过度发散）
    for rank, hit in enumerate(kg_hits):
        pg_id = hit.get("pg_id", 0)
        if pg_id:
            rrf_score = 1.0 / (k + rank + 1) * kg_weight
            scores[pg_id] = scores.get(pg_id, 0) + rrf_score

    # ── 按融合分降序排序 ──
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return sorted_ids
