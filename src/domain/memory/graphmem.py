"""图增强记忆：基于 Neo4j 的记忆图扩展召回。

=============================================================================
                    🧠 记忆系统 第三层：图增强记忆（GraphMemory）
=============================================================================

三层记忆架构的最上层：
  ShortTerm  — "刚才说了什么"      （滑动窗口，进程内，不持久化）
  LongTerm   — "以前聊过什么"      （语义向量 + TF 降级，三阶段合并淘汰）← 基础层
  GraphMemory— "哪些记忆有关联"    （Neo4j FOLLOWS/SIMILAR_TO/CAUSES/BELONGS_TO）← 这里
  Preference — "用户是谁 / 喜欢什么"（KV 画像）

=============================================================================
                    🕸️ 为什么需要在 LTM 之上叠加图结构？
=============================================================================

LTM 的局限：召回是"线性"的
  LTM 通过 embedding 余弦相似度做召回——"找和 query 语义最近的 N 条记忆"。
  但记忆之间是有关系的，LTM 只看到了"点和 query 的距离"，没看到"点和点之间的连线"。

图增强的价值：召回是"跳跃式"的
  例如 LTM 召回了记忆 A："用户问了 Python 异步问题"
  GraphMemory 通过 Neo4j 查询发现 A 有 FOLLOWS 关系的 B："用户问了 FastAPI 框架"
  虽然 B 本身和当前 query 的余弦相似度不够高，但 B 和 A 是先后发生的→相关！
  于是 B 也被加入召回结果——这叫"图扩展召回"。

实际场景举例：
  用户之前问了 3 个 Python 问题的记忆通过 FOLLOWS 关系连成链。
  当前用户问 "Django 怎么部署"，LTM 直接召回了其中 1 条。
  GraphMemory 沿 FOLLOWS 关系扩展出另外 2 条——LLM 看到完整的上下文链，
  回答更连贯："你之前问了 A、B、C，看起来你在学 Python Web 开发…"

=============================================================================
                          🔗 四种关系类型
=============================================================================

Neo4j 中记忆节点之间的关系（Cypher 中的关系类型）：

  (m1)-[:FOLLOWS]->(m2)
    时序关系：m1 发生后，紧接着 m2 发生了。
    在 index_memory() 中自动建立——每次新增记忆时，
    如果传了 prev_item（上一条记忆），就自动创建 FOLLOWS 关系。

  (m1)-[:SIMILAR_TO]-(m2)
    语义相似关系：两条记忆内容相似但不完全相同。
    在 index_similar_relation() 中手动建立——通常在 LTM consolidate
    发现 cosine >= 0.80 时调用，把"相似"关系同步到 Neo4j。

  (m1)-[:CAUSES]->(m2)
    因果关系：m1 导致了 m2。
    预留——当前代码中有查询但未建立，未来由 LLM 分析因果时使用。

  (m1)-[:BELONGS_TO]-(m2)
    归属关系：m1 属于 m2 的一部分/子话题。
    预留——当前代码中有查询但未建立，未来按话题归类时使用。

=============================================================================
                     📊 召回流程：LTM 基础召回 → Neo4j 图扩展
=============================================================================

recall_by_filter() 的两步流程：

  ① LTM 基础召回
     调用 self._ltm.recall_by_filter(query, query_embedding, flt)
     → 得到 base 列表（最多 top_k 条，按 cosine×0.7 + importance×0.3 排序）

  ② Neo4j 图扩展（关键步骤）
     对 base 中的每条记忆：
       Cypher: MATCH (m:Memory {mem_id: X})-[r:FOLLOWS|SIMILAR_TO|CAUSES|BELONGS_TO]-(n:Memory)
               RETURN n.mem_id, type(r) LIMIT 5
       → 找到所有与 X 有关系的邻居节点
       → 收集邻居节点的 mem_id
     → 从 LTM._items 中按 ID 查找这些邻居
     → 追加到 base 列表（去重：排除已在 base 中的）

  结果：base(语义相关) + neighbors(图关联) → LLM 看到更完整的记忆上下文

=============================================================================
                     🏗️ 记忆节点在 Neo4j 中的存储
=============================================================================

Neo4j 节点（Memory label）：
  {
    mem_id: 42,              ← 与 LTM Item.id 一一对应
    content: "用户问了Pyt...", ← 截断到 200 字符（节省 Neo4j 空间）
    importance: 0.75,
    category: "episodic"
  }

注意：content 截断到 200 字符，因为 Neo4j 中的 content 只用于"图可视化时预览"，
不参与文本搜索。完整的 content 在 LTM Item 和 PG 中。Neo4j 只负责图关系。

=============================================================================
                     🛡️ 优雅降级：Neo4j 不可用时完全透明
=============================================================================

整个 GraphMemory 的所有方法都以 available() 检查开头：
  if not self.available(): return ...

当 Neo4j 不可用时：
  index_memory()       → 跳过（节点和关系不创建）
  recall_by_filter()   → 直接返回 LTM 召回结果（无图扩展）
  index_similar_relation() → 跳过
  merge_graph_nodes()  → 跳过
  delete_memory_node() → 跳过

所有 Neo4j 异常都被 try/except 静默捕获（pass），不会向上抛出。
这让图增强成为一个"可选的加分项"——有它更好，没它也行。

这就是为什么 GraphMemory 在构造时必须传入 ltm 参数——
让它知道自己依赖 LTM，没有 LTM 就什么也做不了。
=============================================================================
"""

import threading
from typing import List, Optional, Dict, Set
from src.domain.memory.longterm import LongTerm, Item, RecallFilter


class GraphMemory:
    """
    图增强记忆：通过 Neo4j 客户端建立记忆节点与关系边。

    ── 核心能力 ──
    ① recall_by_filter()   — LTM 召回 + Neo4j 图扩展（召回路径）
    ② index_memory()       — 在 Neo4j 中创建记忆节点 + FOLLOWS 关系（写入路径）
    ③ index_similar_relation() — 建立 SIMILAR_TO 关系（去重时）
    ④ merge_graph_nodes()  — LTM consolidate 合并节点时同步到 Neo4j
    ⑤ delete_memory_node() — 删除节点（淘汰时）

    ── 设计模式：装饰器/代理（Decorator Pattern） ──
    GraphMemory 包装了 LTM，对外暴露相同的 recall_by_filter() 接口。
    调用方（RecallSource）不关心底层是 LTM 还是 GraphMemory——它只需调
    recaller.recall_by_filter()，具体走哪条路径由构造时注入决定：

      # 图可用 → 图增强召回
      RecallSource(self._mem.graph_mem)
      # 图不可用 → 纯 LTM 召回
      RecallSource(self._mem.ltm)

    这是"策略模式"的体现——通过依赖注入切换实现，调用方无感知。

    ── Neo4j 连接 ──
    self._neo4j: 启动时在 main.py 中通过 neo4j_connect() 获取。
    如果连接失败则 _neo4j = None，整个 GraphMemory 静默降级。

    ── 线程安全 ──
    self._mu: threading.RLock() 保护对 LTM._items 的访问
    但 Neo4j Session 本身是线程安全的（每个方法创建独立的 Session）
    """

    def __init__(self, neo4j_client, ltm: LongTerm):
        """
        Args:
            neo4j_client: Neo4j 客户端对象（来自 infrastructure/platform/adapters）。
                          必须有 .available 属性和 .Session() 方法。
                          如果 Neo4j 不可用，传入 None。
            ltm:          长期记忆实例。GraphMemory 依赖 LTM 做基础召回，
                          自己负责在基础召回之上做图扩展。
                          这与 KGStore 不同——KGStore 是 RAG 文档实体的知识图谱，
                          GraphMemory 是记忆节点的关系图。
        """
        self._mu = threading.RLock()
        self._neo4j = neo4j_client
        self._ltm = ltm

    def available(self) -> bool:
        """
        检查 Neo4j 是否可用。

        双重检查：
          ① _neo4j is not None  — 客户端存在
          ② _neo4j.available     — 连接正常（由 infrastructure 层设置）

        所有公开方法都以这个检查开头，不可用时静默跳过。
        """
        return self._neo4j is not None and self._neo4j.available

    # ── 召回路径 ──

    def recall_by_filter(self, query: str, query_embedding: Optional[List[float]],
                         flt: RecallFilter) -> List[Item]:
        """
        图扩展召回：先做 LTM 语义召回，再通过 Neo4j 扩展关联节点。

        ── 两步流程 ──
        ① LTM 基础召回：
           self._ltm.recall_by_filter(query, query_embedding, flt)
           → base 列表（按综合分排序，最多 top_k 条）

        ② Neo4j 图扩展：
           对 base 中每条记忆，查询 Neo4j：
             MATCH (m:Memory {mem_id: X})-[r:REL_TYPE]-(n:Memory)
             沿 FOLLOWS / SIMILAR_TO / CAUSES / BELONGS_TO 四种关系，
             找到关联的邻居节点（每节点最多返回 5 个邻居）。

           从 LTM._items 中按 ID 查找这些邻居 → 追加到 base（去重）。

        ── 图扩展的价值 ──
        LTM 只返回"语义最接近"的 TopK，但可能漏掉"相关但不相似"的记忆。
        例如用户之前讨论了"Python 异步"和"FastAPI"，两个话题 embedding 距离不一定近，
        但它们在 Neo4j 中有 FOLLOWS 关系（同一次对话中先后讨论的）。
        图扩展能找回这种"结构相关但语义不完全一致"的记忆。

        ── 优雅降级 ──
        Neo4j 不可用 → 直接返回 LTM 召回结果（base），跳过图扩展。
        Neo4j 查询异常 → 静默捕获，返回 base。
        无论发生什么，调用方至少能拿到 LTM 的基础结果。

        Args:
            query:            查询文本
            query_embedding:  查询向量（传给 LTM 做语义召回）
            flt:              过滤条件（分类/标签/分数等）

        Returns:
            base + 图扩展邻居 的合并列表（去重后的 Item 列表）
        """
        # ── 第一步：LTM 基础召回 ──
        base = self._ltm.recall_by_filter(query, query_embedding, flt)

        # Neo4j 不可用或基础召回为空 → 直接返回
        if not self.available() or not base:
            return base

        # ── 第二步：Neo4j 图扩展 ──
        try:
            expanded_ids: Set[int] = set()
            sess = self._neo4j.Session()
            try:
                for item in base:
                    # 查找与 item 有关联关系的邻居节点
                    # (m)-[r]-(n) 是无方向的——不管关系箭头指向谁，都算关联
                    records = sess.run(
                        """MATCH (m:Memory {mem_id: $mem_id})-[r:FOLLOWS|SIMILAR_TO|CAUSES|BELONGS_TO]-(n:Memory)
                           RETURN n.mem_id AS mem_id, type(r) AS rel_type
                           LIMIT 5""",
                        {"mem_id": item.id}
                    )
                    for rec in records:
                        nid = rec.get("mem_id")
                        if nid is not None:
                            expanded_ids.add(int(nid))
            finally:
                sess.close()  # 确保回收连接

            # 从 LTM 中查找扩展节点 → 追加到结果
            if expanded_ids:
                with self._ltm._mu:
                    for it in self._ltm._items:
                        if it.id in expanded_ids and it not in base:
                            base.append(it)
        except Exception:
            pass  # Neo4j 异常时静默降级，不影响主流程

        return base

    # ── 写入路径 ──

    def index_memory(self, item: Item, prev_item: Optional[Item] = None):
        """
        在 Neo4j 中为记忆条目建立节点和 FOLLOWS 关系。

        调用时机：
          core_agent._store_long_term_memory() 中，LTM 新增记忆后调用。
          ① 先调 ltm.store_classified() 获取 Item（已分配 ID）
          ② 再调 graph_mem.index_memory(item, prev_item) 同步到 Neo4j

        ── Cypher 操作详解 ──
        ① MERGE 记忆节点（节点标签 Memory，属性 mem_id）
           MERGE 语义：如果 mem_id 已存在则更新属性，否则创建新节点。
           属性：content（截断到 200 字符）、importance、category。
           截断原因：Neo4j 只用于图关系查询，完整内容在 LTM 和 PG 中。

        ② MERGE FOLLOWS 关系
           如果传了 prev_item（上一条记忆），则创建 (prev)→(curr) 的 FOLLOWS 关系。
           MERGE 语义：如果关系已存在则复用，不存在则创建。
           这样多次调用不会创建重复的 FOLLOWS 边。

        ── 优雅降级 ──
        Neo4j 不可用 → 直接 return，不报错。
        Neo4j 异常 → try/except 捕获，静默跳过。

        Args:
            item:      要创建节点的记忆条目（必须有有效的 item.id）
            prev_item: 上一条记忆（可选），用于建立 FOLLOWS 时序关系
        """
        if not self.available():
            return

        try:
            sess = self._neo4j.Session()
            try:
                # ── ① 创建/更新记忆节点 ──
                #     只存前 200 字符的内容预览 + 重要性 + 分类
                sess.run(
                    """MERGE (m:Memory {mem_id: $mem_id})
                       SET m.content = $content, m.importance = $importance,
                           m.category = $category""",
                    {"mem_id": item.id, "content": item.content[:200],
                     "importance": item.importance, "category": item.category}
                )
                # ── ② 建立 FOLLOWS 关系 ──
                #     如果有上一条记忆，建立时序边
                if prev_item:
                    sess.run(
                        """MATCH (a:Memory {mem_id: $prev_id}), (b:Memory {mem_id: $curr_id})
                           MERGE (a)-[r:FOLLOWS]->(b)""",
                        {"prev_id": prev_item.id, "curr_id": item.id}
                    )
            finally:
                sess.close()
        except Exception:
            pass  # Neo4j 异常静默降级

    def delete_memory_node(self, mem_id: int):
        """
        删除 Neo4j 中的记忆节点（及其所有关系边）。

        调用时机：
          LTM consolidate Phase 3 过期淘汰后，core_agent 同步删除 PG 中的条目，
          同时调用此方法删除 Neo4j 中的对应节点。

        Cypher: DETACH DELETE — 先断开所有关系边，再删除节点本身。
        如果不用 DETACH，有关系的节点无法直接删除（Neo4j 的保护机制）。

        Args:
            mem_id: 要删除的记忆 ID（对应 LTM Item.id）
        """
        if not self.available():
            return
        try:
            sess = self._neo4j.Session()
            try:
                sess.run("MATCH (m:Memory {mem_id: $mem_id}) DETACH DELETE m",
                         {"mem_id": mem_id})
            finally:
                sess.close()
        except Exception:
            pass

    def index_similar_relation(self, item_a: Item, item_b: Item, similarity: float):
        """
        建立 SIMILAR_TO 关系（双向）。

        调用时机：
          LTM consolidate Phase 2 中发现两条记忆的 cosine 相似度 >= 0.80，
          但又不完全重复（< 0.95）时，在 Neo4j 中记录它们的相似关系。

        MERGE 语义：如果两个节点之间已有 SIMILAR_TO 边则更新 score，否则创建。

        Args:
            item_a:     记忆 A
            item_b:     记忆 B
            similarity: 余弦相似度 score（0.80 ~ 0.95），存储为关系属性
        """
        if not self.available():
            return
        try:
            sess = self._neo4j.Session()
            try:
                sess.run(
                    """MATCH (a:Memory {mem_id: $id_a}), (b:Memory {mem_id: $id_b})
                       MERGE (a)-[r:SIMILAR_TO]->(b)
                       SET r.score = $score""",
                    {"id_a": item_a.id, "id_b": item_b.id, "score": similarity}
                )
            finally:
                sess.close()
        except Exception:
            pass

    def merge_graph_nodes(self, keep_id: int, remove_ids: List[int]):
        """
        合并节点：将 remove_ids 的关系转移给 keep_id 后删除旧节点。

        调用时机：
          LTM consolidate Phase 2 合并记忆后，core_agent 调此方法同步 Neo4j。

        ── 操作步骤 ──
        ① 对每个 remove_id：
           找到 old 节点的关系边（排除指向 keep_id 的边），删除这些边。
           注意：只删边不删节点——这样 old 和 n 之间的连接就断了。
        ② 删除 old 节点自身（DETACH DELETE）
           如果还有残留边（没在第①步删干净的），DETACH 自动处理。

        ── 为什么不把关系重新连接到 keep_id？ ──
        当前实现是先删边再删节点，没有做关系迁移。
        这意味着合并后，旧节点的关系会丢失。
        这是一个简化实现——要完整迁移需要更复杂的 Cypher 查询。
        对于记忆场景，丢失几个 FOLLOWS 边的影响很小（LTM 才是关系的主要来源）。

        Args:
            keep_id:    保留的节点 ID
            remove_ids: 需要删除的节点 ID 列表
        """
        if not self.available():
            return
        try:
            sess = self._neo4j.Session()
            try:
                for rid in remove_ids:
                    # ① 删除旧节点与邻居的关系边（排除指向 keep_id 的边）
                    sess.run(
                        """MATCH (old:Memory {mem_id: $old_id})-[r]-(n)
                           WHERE n.mem_id <> $keep_id
                           DELETE r""",
                        {"old_id": rid, "keep_id": keep_id}
                    )
                    # ② 删除旧节点自身
                    sess.run(
                        "MATCH (m:Memory {mem_id: $old_id}) DETACH DELETE m",
                        {"old_id": rid}
                    )
            finally:
                sess.close()
        except Exception:
            pass
