"""长期记忆：支持语义向量召回（embedding 优先）或 TF 词袋降级。

=============================================================================
                    🧠 记忆系统 第二层：长期记忆（LTM）
=============================================================================

三层记忆架构中的核心层：
  ShortTerm  — "刚才说了什么"（滑动窗口，进程内，不持久化）
  LongTerm   — "以前聊过什么（语义）"（embedding + TF 双路召回，持久化到 PG）← 这里
  GraphMemory— "哪些记忆有关联"（Neo4j FOLLOWS/SIMILAR_TO 图扩展）
  Preference — "用户是谁"（KV 画像，LLM + 规则提取）

=============================================================================
                           📦 数据结构：Item
=============================================================================

每条长期记忆是一个 Item 对象，字段如下：

  Item {
    id            → 自增 ID（PG 主键，跨进程唯一标识，持久化后不变）
    content       → 记忆文本，如 "用户问了Python异步编程问题"
    importance    → 重要性 [0, 1]。LLM 对关键信息打高分（如"用户偏好简洁回答"），
                    闲聊打低分。用于召回加权和淘汰判断（Phase 3）
    embedding     → 语义向量（List[float]），2560 维（豆包）或 3072 维（OpenAI），
                    由 LLM embedding API 生成，用于余弦相似度计算
    created_at    → 创建时间戳（Unix 秒数），用于衰减和过期淘汰
    last_accessed → 最后被召回的访问时间（预留给 LRU 淘汰）
    category      → 记忆分类，影响召回时的分类过滤：
                    "episodic"     — 事件性记忆（"用户上次问了X"）
                    "fact"         — 事实性记忆（"Python 是动态语言"）
                    "user_info"    — 用户个人信息（"姓名张三"、"职业后端开发"）
                    "general"      — 通用类
                    "tool_failure" — 工具失败教训（ReAct 模式特殊召回）
    tags          → 自由标签列表，如 ["python", "async", "bug"]
    slot_hint     → 指定槽位类型（"profile" / "task" / ""），指示放入哪个提示词槽位
    score         → 综合召回分（运行时计算，不持久化）= sim×0.7 + importance×0.3
    tf_vector     → TF 词袋向量（Dict[str, float]），embedding 不可用时的降级路径
  }

=============================================================================
                        🔍 双路召回机制
=============================================================================

路径 1（主路径）：Embedding 语义召回
  ① 用 LLM API 把 query 向量化 → query_embedding（2560 维 float list）
  ② 遍历所有 Item，计算 cosine(query_embedding, item.embedding)
  ③ score = cos_sim × 0.7 + importance × 0.3  （语义相关度主导，重要性调权）
  ④ 过滤 score < min_score → 按 score 降序排列 → 截断取 top_k

路径 2（降级路径）：TF 词袋召回
  → 当 LLM API 不可用或未传 query_embedding 时自动触发
  ① 把 query 文本分词 → {词: 词频}（如 "Python异步编程" → {"Python": 0.33, "异步": 0.33, ...}）
  ② 对每个 Item.tf_vector 计算稀疏余弦相似度
  ③ score = tf_cos × 0.7 + importance × 0.3 → 过滤 → 排序 → 截断

为什么需要 TF 降级？
  LLM embedding API 可能限流、超时、不可用。TF 词袋虽然不如语义向量准确，
  但零依赖、零成本、零延迟，保证记忆召回在最坏情况下也能工作（优雅降级）。

为什么 score 公式是 0.7:0.3 而不是各 0.5？
  语义相关性（0.7）是主导因素——先要"相关"，然后才比较"重要"。
  重要性（0.3）是调权因素——同等相关时，重要的优先呈现。

=============================================================================
                       📊 三阶段合并淘汰（Consolidate）
=============================================================================

触发条件：_store_count >= MemoryConsolidationTrigger（config 中配置，默认 10）
含义：上次合并后又新增了 10 条记忆 → 该做一次清理了。

Phase 1: 重要性衰减
  公式：importance *= DecayRate ^ days
  其中 DecayRate = 0.995（默认），days = (now - created_at) / 86400

  衰减曲线：
    days=1   → importance *= 0.995^1   ≈ 0.995  （几乎不变）
    days=7   → importance *= 0.995^7   ≈ 0.965  （略降）
    days=30  → importance *= 0.995^30  ≈ 0.860  （稳定下降）
    days=100 → importance *= 0.995^100 ≈ 0.606  （显著下降）

  为什么是指数衰减而不是线性？
    指数衰减更符合人类遗忘规律（艾宾浩斯遗忘曲线）。
    刚发生的事遗忘快，剩下来的记忆衰减越来越慢。

Phase 2: 去重 + 合并（双重阈值）
  双重循环 O(n²) 比较所有 Item 对：

  阈值 A: cosine >= DedupThreshold (0.95) → 完全重复
    → 只保留 importance 更高的，标记另一个删除

  阈值 B: cosine >= SimilarityThreshold (0.80) → 语义相似但不完全重复
    → 保留较长内容的 item，吸收较短内容的 importance（取 max）

  为什么去重是 O(n²) 而不是哈希？
    这里的"重复"是语义重复（cosine 相似度高），不是文本相同。
    无法用哈希快速查找——只能用向量计算每对比较。
    n 通常 < 1000，O(n²) 可接受。

Phase 3: 过期淘汰
  淘汰条件（两个同时满足才淘汰）：
    ① (now - created_at) > TTL（默认 30 天）
    ② importance < MinImport（默认 0.3）

  含义：重要记忆永久保留，不重要的过期删除。
  高重要性记忆（>0.3）即使超过 30 天也不删除——"重要的事情不会忘"。

合并后清理：
  ① 批量删除被标记的 Item 索引
  ② _rebuild_vocab() — 用剩余记忆重建全局词汇表和所有 TF 向量
  ③ _store_count = 0 — 重置计数器，等下一次触发

=============================================================================
                      🔗 与 PG 持久化的双向同步
=============================================================================

写入：store_classified() → 返回 Item → core_agent 调 PG repo.save() 持久化
恢复：启动时 PG SELECT * FROM long_term_memory → store_item() 逐条恢复到内存
合并：consolidate() → 内存删除/更新 → core_agent 调 PG repo 同步

embedding 在 PG 中以 JSON TEXT 列存储（如 "[0.123, -0.456, ...]"），
因为 PG 中不需要做向量运算——向量搜索全在 Python 内存中做 cosine 计算。
PG 只负责"关了程序下次还能读回来"的持久化职责。
=============================================================================
"""

import math
import threading
import time
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from collections import Counter


# ═══════════════════════════════════════════════════════════════════════════
#  数据结构
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Item:
    """
    单条长期记忆。

    字段详解：
      id            — 全局唯一自增 ID。负数 = 未分配（store_item 会分配）。
                      持久化到 PG 时作为主键，跨进程唯一标识。
      content       — 记忆正文，如 "用户喜欢用 Python 做自动化脚本"
      importance    — 重要性 [0, 1]。LLM 提取记忆时打分：关键信息 > 0.7，普通 > 0.5，闲聊 < 0.3。
                      用于：召回加权（score 公式）、Phase 3 过期淘汰判断。
      embedding     — 语义向量。2560 维（豆包/火山引擎）或 3072 维（OpenAI text-embedding-3-large）。
                      由 LLM embedding API 生成，用余弦相似度衡量语义接近程度。
                      空列表 = 还没 embedding（可能是暂时没调用 API，后续补充）。
      created_at    — 创建时间戳（Unix 秒数）。用于衰减天数计算和 TTL 过期判断。
      last_accessed — 最后被召回的时间。预留给 LRU 淘汰策略使用（当前未严格实现，保留字段）。
      category      — 记忆分类。不是随意字符串，而是预定义集合：
                      "episodic"     — 事件性（对话情节记忆）
                      "fact"         — 事实性（客观事实）
                      "user_info"    — 用户信息（个人资料类）
                      "general"      — 通用
                      "tool_failure" — 工具失败教训（ReAct 模式下可单独召回这类记忆）
                      分类影响召回时的过滤——不同 Schema 会指定不同的 categories。
      tags          — 自由标签列表，如 ["python", "async", "debug"]。
                      召回时可通过 require_tags 做 AND 过滤。
      slot_hint     — 指定槽位。如 "profile" 表示这条记忆属于用户画像槽位。
                      不为空时，RecallSource 可据此定向分配，提高召回准确性。
      score         — 综合召回分 = cos_sim×0.7 + importance×0.3。
                      每次召回时重新计算并写入，不持久化——不同查询得分不同。
      tf_vector     — TF 词袋向量（如 {"python": 0.33, "异步": 0.33, "编程": 0.33}）。
                      embedding 不可用时用于降级召回。在 store_classified() 时构建，
                      consolidate() 后整体重建。
    """
    id: int = -1
    content: str = ""
    importance: float = 0.5
    embedding: List[float] = field(default_factory=list)
    created_at: float = 0.0
    last_accessed: float = 0.0
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    slot_hint: str = ""
    score: float = 0.0            # 综合召回分 = cos_sim × 0.7 + importance × 0.3
    tf_vector: Dict[str, float] = field(default_factory=dict)


@dataclass
class RecallFilter:
    """
    召回时的过滤条件 — 从 Schema 的 SlotFilter 映射而来。

    数据流：
      Schema 中定义 Slot.filter（如 categories=["episodic","fact"], top_k=3, min_score=0.4）
        → RecallSource.fetch() 翻译为 RecallFilter 对象
          → LongTerm.recall_by_filter() 按条件执行过滤和召回

    字段说明：
      categories    — 限定分类白名单，如 ["episodic", "fact"]。空列表 = 不限分类。
      require_tags  — 限定标签，要求 item.tags 包含全部 require_tags（AND 关系）。
      min_score     — 最低综合分数阈值，低于此分的记忆不返回。默认 0.4。
                      设太低 → 质量差的记忆也返回；设太高 → 可能没有结果。
      top_k         — 返回数量上限，按分数从高到低取前 N 条。
      max_age_hours — 时间窗口（小时），只召回 N 小时内的记忆。0 = 不限时间。
    """
    categories: List[str] = field(default_factory=list)
    require_tags: List[str] = field(default_factory=list)
    min_score: float = 0.4
    top_k: int = 3
    max_age_hours: int = 0


# ═══════════════════════════════════════════════════════════════════════════
#  长期记忆类
# ═══════════════════════════════════════════════════════════════════════════

class LongTerm:
    """
    长期记忆：支持语义向量 / TF 词袋双层召回。

    ── 核心职责 ──
    ① store_classified() — 写入记忆（自动向量去重）
    ② recall_by_filter()  — 条件召回（embedding 优先，TF 降级）
    ③ consolidate()       — 三阶段合并淘汰
    ④ need_consolidation()— 判断是否该触发合并

    ── 存储结构 ──
    self._items: List[Item]
      纯 Python 列表。同时存储 embedding 和 tf_vector，以空间换可靠性。

    self._vocab: Dict[str, int]
      全局词汇表，{词: 整数索引}。用于 TF 向量构建（_build_tf_for 注册新词）。
      注意：不存 IDF，只存索引——降级路径不需要 IDF（记忆数 < 1000，IDF 价值有限）。

    self._store_count: int
      自上次合并以来新增的记忆数。达到 trigger 阈值时触发 consolidate()。

    self._next_id: int
      自增 ID 计数器，从 1 开始。每次新建 Item 时 +1。

    ── 线程安全 ──
    threading.RLock() 保护所有读写操作：
      • store_classified() — core_agent 后处理线程调用
      • recall_by_filter() — promptctx assemble 线程调用
      • consolidate()      — 合并淘汰线程调用
      三者可能并发，需要锁保护。

    ── 与 PG 持久化的协作 ──
    写入：store_classified() → 返回 Item → core_agent 调 PG repo.save(item) 持久化
    恢复：启动 → PG SELECT → store_item(item) → 逐条恢复到 self._items
    合并：consolidate() → core_agent 调 PG repo 同步删除/更新被淘汰的记忆
    """

    def __init__(self, cfg=None):
        """
        Args:
            cfg: 配置对象（config.yaml 中的 Memory 段落），包含：
                 MemoryConsolidationTrigger   — 触发合并的新增数阈值（默认 10）
                 MemoryConsolidationDecayRate — 每日衰减率（默认 0.995）
                 MemoryConsolidationDedup     — 完全重复去重阈值（默认 0.95）
                 MemoryConsolidationSimilarity— 相似合并阈值（默认 0.80）
                 MemoryConsolidationTTLDays   — 过期天数（默认 30）
                 MemoryConsolidationMinImport — 过期淘汰的最低重要性（默认 0.3）

                 配置命名约定：MemoryConsolidation + 具体参数名（PascalCase）
        """
        self._mu = threading.RLock()
        self._items: List[Item] = []
        self._next_id = 1              # 自增 ID，从 1 开始
        self._vocab: Dict[str, int] = {}  # 全局词汇表
        self._store_count = 0          # 合并后的新增计数
        self._cfg = cfg

    # ────────────────────────────────────────────────────────────────
    #  写入操作
    # ────────────────────────────────────────────────────────────────

    def store_item(self, item: Item):
        """
        直接存储一条记忆——不检查重复，不计算 TF。

        ⚠️ 与 store_classified() 的关键区别：
          store_item()       — 不做 dedup、不计算 TF、保留原始 ID（用于 PG 恢复）
          store_classified() — 先 dedup 检查、分配新 ID、计算 TF、store_count++（用于新增）

        调用时机：
          main.py 启动流程 — PG SELECT * FROM long_term_memory — 遍历结果 — store_item(item)
          从 PG 恢复的记忆已经是去重后的，不需要再检查重复。

        自动处理：
          • 无 ID 时自动分配
          • 有 ID 且 ID >= _next_id 时更新计数器（防止恢复后 ID 冲突）
          • 无时间戳时自动设为当前时间
        """
        with self._mu:
            if item.id < 0:
                # PG 恢复时 ID 已存在，不会走这里；手动创建时自动分配
                item.id = self._next_id
                self._next_id += 1
            else:
                # 从 PG 恢复时 ID 已确定，保证 _next_id 不冲突（比所有已有 ID 都大）
                if item.id >= self._next_id:
                    self._next_id = item.id + 1
            if item.created_at == 0:
                item.created_at = time.time()
            if item.last_accessed == 0:
                item.last_accessed = time.time()
            self._items.append(item)

    def store_classified(
        self, content: str, importance: float = 0.5,
        embedding: Optional[List[float]] = None,
        category: str = "general", tags: List[str] = None,
        slot_hint: str = ""
    ) -> Optional[Item]:
        """
        写入记忆并自动去重——新增记忆的主要入口。

        ── 完整流程 ──
        ① 初始化默认值（tags=[], emb=[]）
        ② 从 config 读取 DedupThreshold（默认 0.95）
        ③ 如果有 embedding → 遍历已有 Item 计算 cosine 相似度
        ④ cosine >= 阈值 → 更新已有条目（importance 取 max，content 刷新），返回已有条目
        ⑤ 未命中 → 新建 Item，分配 ID，追加到 _items，_store_count++

        ── 去重的意义 ──
        用户可能反复说同一句话（如两次对话都说"我喜欢 Python"），
        不希望记忆库中存两条相同的记忆，因此做语义去重。

        为什么 threshold 是 0.95 而不是更低？
          0.95 ≈ 角度 18°——向量几乎同向，语义高度一致。
          设太低（如 0.7）可能把"Python编程"和"Django框架"错误合并。
          宁可漏过重复（O(n²) 比较时性能可接受），也不能错误合并。

        ── 调用时机 ──
        core_agent 每轮对话的 post-processing 阶段：
          _store_long_term_memory(content=会话总结文本, importance=LLM打分, embedding=LLM向量化)

        Args:
            content:    记忆文本内容（由 LLM 从对话中提取的关键信息）
            importance: 初始重要性 [0, 1]，默认 0.5
            embedding:  语义向量（2560 维 float list），可为空
            category:   分类标签，默认 "general"
            tags:       自由标签列表，用于过滤
            slot_hint:  槽位类型提示，如 "profile"

        Returns:
            新建或更新后的 Item。去重命中时返回已有的 Item（store_count 不增加）。
        """
        with self._mu:
            tags = tags or []
            emb = embedding or []

            # ── 从 config 读取去重阈值 ──
            dedup_threshold = 0.95
            if self._cfg:
                dedup_threshold = self._cfg.MemoryConsolidationDedup

            # ── 去重检查：遍历已有记忆做 cosine 比较 ──
            if emb:
                for existing in self._items:
                    if existing.embedding:
                        sim = self._cosine(emb, existing.embedding)
                        if sim >= dedup_threshold:
                            # 语义相同 → 合并而非新增
                            existing.importance = max(existing.importance, importance)
                            existing.last_accessed = time.time()
                            existing.content = content  # 用最新版本覆盖旧内容
                            return existing

            # ── 未命中去重 → 新建 Item ──
            new_item = Item(
                id=self._next_id,
                content=content,
                importance=importance,
                embedding=emb,
                created_at=time.time(),
                last_accessed=time.time(),
                category=category,
                tags=tags,
                slot_hint=slot_hint,
            )
            self._next_id += 1
            self._items.append(new_item)
            self._store_count += 1  # 累计到阈值后触发 consolidate()
            return new_item

    # ────────────────────────────────────────────────────────────────
    #  召回操作
    # ────────────────────────────────────────────────────────────────

    def recall(self, query: str, query_embedding: Optional[List[float]] = None,
               top_k: int = 3) -> List[Item]:
        """
        简单语义召回（不带分类/标签过滤条件）。

        这是 recall_by_filter() 的便捷包装——用默认 RecallFilter（仅设 top_k）。
        适用于不需要复杂过滤的场景（如通用聊天模式快速召回相关记忆）。
        """
        return self.recall_by_filter(query, query_embedding, RecallFilter(top_k=top_k))

    def recall_by_filter(self, query: str, query_embedding: Optional[List[float]],
                         flt: RecallFilter) -> List[Item]:
        """
        条件召回——长期记忆的核心查询方法。

        ── 六步过滤 + 打分流程 ──
        ① 分类过滤：flt.categories 非空 → 只保留 category in categories_set 的条目
        ② 标签过滤：flt.require_tags 非空 → 只保留 tags 包含所有 require_tags 的条目（子集关系）
        ③ 时间过滤：flt.max_age_hours > 0 → 只保留 N 小时内创建的条目
        ④ 语义/词袋打分：
           embedding 可用 → cosine_sim × 0.7 + importance × 0.3
           embedding 不可用 → tf_cos × 0.7 + importance × 0.3
        ⑤ 过滤低分：score < min_score 的条目丢弃
        ⑥ 排序截断：按 score 降序排列，截断到 top_k 条

        ── 为什么过滤在打分之前？ ──
        过滤是 O(n) 的简单条件判断，打分是 O(n×dim) 的向量计算。
        先过滤掉不相关的条目，减少候选集（如从 500 条过滤到 50 条），
        再对剩余 50 条做 cosine 计算，节省不必要的向量运算。

        ── TF 降级模式 ──
        当 query_embedding 为 None 时自动触发：
          ① _text_to_tf(query) → {词: 词频}
          ② 对每个 Item.tf_vector（如果有）计算稀疏余弦相似度
          ③ 若 Item 也没有 tf_vector → sim = 0.0（无法匹配，但仍可因 import 高而通过过滤）

        Args:
            query:            查询文本（如用户当前问题）
            query_embedding:  查询向量（LLM embedding API 生成）；None → 走 TF 降级
            flt:              召回过滤条件（分类/标签/分数/时间/数量限制）

        Returns:
            按综合分降序排列的 Item 列表，最多 top_k 条。
            如果没有任何条目通过过滤，返回空列表 []。
        """
        with self._mu:
            candidates: List[Item] = []

            # ── ① 分类过滤 ──
            #     白名单模式：categories 非空时只保留分类在集合中的条目
            if flt.categories:
                cat_set = set(flt.categories)
                items = [it for it in self._items if it.category in cat_set]
            else:
                items = list(self._items)

            # ── ② 标签过滤 ──
            #     AND 关系：条目标签必须是 require_tags 的超集
            #     如 require_tags=["python","async"] → item.tags=["python","async","bug"] ✅
            #                                         item.tags=["python"] ❌
            if flt.require_tags:
                tag_set = set(flt.require_tags)
                items = [it for it in items if tag_set.issubset(set(it.tags))]

            # ── ③ 时间过滤 ──
            #     限制只召回最近 N 小时内的记忆（用于"最近的讨论"等场景）
            if flt.max_age_hours > 0:
                cutoff = time.time() - flt.max_age_hours * 3600
                items = [it for it in items if it.created_at >= cutoff]

            # 过滤后无候选 → 空结果
            if not items:
                return []

            # ── ④ 语义/词袋打分 ──
            if query_embedding:
                # 路径 1：Embedding 语义相似度（主路径）
                # 对每个有 embedding 的 Item 计算 cosine，综合分 = 相似度×0.7 + 重要性×0.3
                for it in items:
                    if it.embedding:
                        sim = self._cosine(query_embedding, it.embedding)
                        score = sim * 0.7 + it.importance * 0.3
                        if score >= flt.min_score:
                            it.score = score
                            candidates.append(it)
            else:
                # 路径 2：TF 词袋降级（embedding 不可用时）
                query_tf = self._text_to_tf(query)
                for it in items:
                    if it.tf_vector:
                        sim = self._cosine_tf(query_tf, it.tf_vector)
                    else:
                        sim = 0.0  # 既无 embedding 也无 tf → 完全不匹配
                    score = sim * 0.7 + it.importance * 0.3
                    if score >= flt.min_score:
                        it.score = score
                        candidates.append(it)

            # ── ⑤ 排序截断 ──
            #     按 score 降序：最相关的排在前面，LLM 先看到
            candidates.sort(key=lambda x: x.score, reverse=True)
            if flt.top_k > 0:
                candidates = candidates[:flt.top_k]

            # ── ⑥ 更新访问时间 ──
            #     命中的记忆标记为"最近访问"，预留给未来的 LRU 淘汰策略
            for it in candidates:
                it.last_accessed = time.time()

            return candidates

    def filter_by_category(self, categories: List[str], limit: int = 10) -> List[Item]:
        """
        按分类筛选，按重要性排序（不计算语义相似度）。

        与 recall_by_filter() 的区别：
          filter_by_category() — 纯分类过滤 + 重要性排序，不做相似度计算
                                用于"看看有哪些 user_info 类型的记忆"这类管理查询
          recall_by_filter()   — 语义相似度 + 分类过滤 + 重要性排序
                                用于"找和当前话题最相关的记忆"

        调用场景：
          前端"记忆管理"页：按分类浏览所有记忆
        """
        with self._mu:
            cat_set = set(categories)
            filtered = [it for it in self._items if it.category in cat_set]
            filtered.sort(key=lambda x: x.importance, reverse=True)
            return filtered[:limit]

    # ────────────────────────────────────────────────────────────────
    #  合并淘汰
    # ────────────────────────────────────────────────────────────────

    def need_consolidation(self) -> bool:
        """
        检查是否需要触发合并淘汰。

        判断条件：_store_count >= MemoryConsolidationTrigger（默认 10）
        含义：自上次合并以来新增了 10 条记忆 → 该做一次清理了。

        为什么不每条新增都触发？
          consolidate() 是 O(n²) 的（Phase 2 双重循环），每条触发太浪费。
          攒到一定数量批量处理，性能 & 效果平衡。
        """
        if not self._cfg:
            return False
        return self._store_count >= self._cfg.MemoryConsolidationTrigger

    def consolidate(self):
        """
        执行三阶段合并淘汰：衰减 → 去重合并 → 过期淘汰。

        ── 调用链路 ──
        core_agent post-processing
          → if ltm.need_consolidation():
              result = ltm.consolidate()
              # 把 result 同步到 PG（repo.delete + repo.update）

        ── Phase 1: 重要性衰减 ──
        遍历所有记忆，对每条的 importance 应用指数衰减：
          importance *= DecayRate ^ days
        这是记忆系统的"遗忘机制"：不重要的记忆随时间自然淡化。

        ── Phase 2: 去重 + 合并（双重循环 O(n²)） ──
        对所有 (i, j) 对做 cosine 比较：
          ① sim >= DedupThreshold (0.95)：完全重复
             → 保留 importance 更高的，标记另一个删除
          ② sim >= SimilarityThreshold (0.80)：语义相似但不完全相同
             → 保留内容较长的，合并 importance（取 max），标记短的删除

        去重与合并的区别：
          去重(0.95) — "几乎一样" → 删一个，留一个
          合并(0.80) — "说得差不多" → 吸取短的内容进长的，然后删短的

        ── Phase 3: 过期淘汰 ──
        删除同时满足以下两个条件的记忆：
          ① 超过 TTL 天数（默认 30 天）
          ② importance < MinImport（默认 0.3）
        重要记忆（>0.3）永久保留，不受 30 天限制。

        ── 合并后清理 ──
        ① 批量删除被标记的条目
        ② _rebuild_vocab()：用剩余记忆重建全局词汇表 + 所有 TF 向量
           （因为合并删除了条目，词汇统计变了）
        ③ _store_count = 0：重置计数器

        Returns:
            dict: {
                "deleted_ids":  [int, ...],              # PG DELETE 全部被删 ID
                "merged_pairs": {keep_id: [remove_id]},  # Neo4j 合并关系迁移
                "evicted_ids": [int, ...],               # Neo4j 直接删除节点
                "updated_items": [Item, ...],            # PG UPDATE 内容/重要性变化
            }
            调用方负责将结果同步到 PG 和 Neo4j。
        """
        with self._mu:
            if not self._cfg:
                return {"deleted_ids": [], "merged_pairs": {}, "evicted_ids": [], "updated_items": []}

            cfg = self._cfg
            now = time.time()
            deleted_ids: List[int] = []
            updated_items: List[Item] = []

            # ── Phase 1: 重要性衰减 ──
            #     指数衰减：importance = importance × decayRate ^ days
            #     衰减率 0.995 意味着 100 天后重要性降到 60%
            for it in self._items:
                old_imp = it.importance
                days = (now - it.created_at) / 86400
                it.importance *= cfg.MemoryConsolidationDecayRate ** days
                # 重要性变化超过 1% → 标记为需要 PG 更新
                if abs(it.importance - old_imp) > 0.001:
                    updated_items.append(it)

            # ── Phase 2: 去重 + 合并（双重循环） ──
            #     两个阈值：
            #       0.95 — 完全重复，去重
            #       0.80 — 语义相似，合并
            #
            #     merged_pairs 记录 (保留ID → [被合并ID, ...])，用于 Neo4j 关系迁移
            removed_indices: set = set()
            merged_pairs: Dict[int, List[int]] = {}  # keep_id → [remove_id, ...]
            n = len(self._items)
            for i in range(n):
                if i in removed_indices:
                    continue
                for j in range(i + 1, n):
                    if j in removed_indices:
                        continue
                    a, b = self._items[i], self._items[j]
                    if a.embedding and b.embedding:
                        sim = self._cosine(a.embedding, b.embedding)
                        if sim >= cfg.MemoryConsolidationDedup:
                            # 完全重复 → 保留 importance 更高的
                            if a.importance >= b.importance:
                                removed_indices.add(j)
                                merged_pairs.setdefault(a.id, []).append(b.id)
                            else:
                                removed_indices.add(i)
                                merged_pairs.setdefault(b.id, []).append(a.id)
                                break
                        elif sim >= cfg.MemoryConsolidationSimilarity:
                            # 语义相似 → 合并（保留较长内容，取高重要性）
                            content_changed = len(b.content) > len(a.content)
                            imp_changed = b.importance > a.importance
                            if content_changed:
                                a.content = b.content
                            a.importance = max(a.importance, b.importance)
                            if content_changed or imp_changed:
                                updated_items.append(a)
                            removed_indices.add(j)
                            merged_pairs.setdefault(a.id, []).append(b.id)

            # 收集 Phase 2 被删条目对应的 Item ID
            phase2_deleted: List[int] = []
            if removed_indices:
                for idx in removed_indices:
                    phase2_deleted.append(self._items[idx].id)
                deleted_ids.extend(phase2_deleted)
                self._items = [it for i, it in enumerate(self._items) if i not in removed_indices]

            # ── Phase 3: 过期淘汰 ──
            #     两个条件同时满足才淘汰：超时 AND 不重要
            ttl_seconds = cfg.MemoryConsolidationTTLDays * 86400
            evicted = [
                it for it in self._items
                if (now - it.created_at) > ttl_seconds and it.importance < cfg.MemoryConsolidationMinImport
            ]
            evicted_ids = [it.id for it in evicted]
            deleted_ids.extend(evicted_ids)

            self._items = [
                it for it in self._items
                if not ((now - it.created_at) > ttl_seconds and it.importance < cfg.MemoryConsolidationMinImport)
            ]

            # ── 去重：被删除的条目不需要再 UPDATE ──
            deleted_set = set(deleted_ids)
            updated_items = [it for it in updated_items if it.id not in deleted_set]

            # ── 合并后重建 TF 词表和词汇表 ──
            self._rebuild_vocab()
            self._store_count = 0

            return {
                "deleted_ids": deleted_ids,       # PG DELETE: 全部被删 ID
                "merged_pairs": merged_pairs,     # Neo4j merge_graph_nodes: {keep_id: [remove_id, ...]}
                "evicted_ids": evicted_ids,       # Neo4j delete_memory_node: 过期淘汰 ID
                "updated_items": updated_items,   # PG UPDATE: 内容/重要性变化条目
            }

    # ────────────────────────────────────────────────────────────────
    #  工具方法
    # ────────────────────────────────────────────────────────────────

    def count(self) -> int:
        """返回当前记忆总数（日志/调试用）"""
        with self._mu:
            return len(self._items)

    def snapshot(self) -> List[Item]:
        """
        返回所有记忆的副本（线程安全）。

        调用时机：
          持久化到 PG 时获取全量列表
          前端"记忆管理"页面展示全量数据
        """
        with self._mu:
            return list(self._items)

    # ────────────────────────────────────────────────────────────────
    #  TF 词袋构建（内部方法）
    # ────────────────────────────────────────────────────────────────

    def _rebuild_vocab(self):
        """
        重建全局词汇表和所有 Item 的 TF 向量。

        调用时机：
          ① consolidate() 合并淘汰后——Item 列表变了，旧 TF 过期
          ② 从 PG 恢复全量记忆后（通过外部调用或手动触发）

        流程：
          ① _vocab 清空
          ② 遍历所有 Item → _build_tf_for(item)
             → 每个 item 会注册新词到 _vocab 并计算 item.tf_vector
        """
        self._vocab = {}
        for it in self._items:
            self._build_tf_for(it)

    def _build_tf_for(self, item: Item):
        """
        为单条记忆构建 TF（词频）向量，结果写入 item.tf_vector。

        TF 公式：tf(word) = 该词在本文中的出现次数 / 文本总词数

        示例：
          content = "Python 异步 Python 编程"
          → words = ["python", "异步", "python", "编程"]
          → total = 4
          → counts = Counter → {"python": 2, "异步": 1, "编程": 1}
          → tf_vector = {"python": 0.5, "异步": 0.25, "编程": 0.25}

        同时：把新词注册到全局 _vocab（分配一个整数索引）。
        _vocab 只存词和索引，不存 IDF——记忆数通常 < 1000，IDF 区分度有限。

        ⚠️ 分词限制：
        这里用 .split() 做英文分词，对中文效果差（一个中文字算一个"词"）。
        理想情况下应该引入 jieba 等中文分词工具，但作为降级路径，.split() 足够。
        """
        words = item.content.lower().split()
        if not words:
            return
        tf: Dict[str, float] = {}
        # 注册新词到全局词汇表（分配整数索引）
        for w in words:
            if w not in self._vocab:
                self._vocab[w] = len(self._vocab)
        total = len(words)
        counts = Counter(words)
        for w, c in counts.items():
            tf[w] = c / total
        item.tf_vector = tf

    def _text_to_tf(self, text: str) -> Dict[str, float]:
        """
        把查询文本转为 TF 词袋向量（临时计算，不更新 _vocab）。

        与 _build_tf_for() 的区别：
          _build_tf_for() — 写入 Item.tf_vector，同时更新全局 _vocab
          _text_to_tf()   — 只返回 TF Dict，不修改全局状态（临时查询用）

        注意：这里不依赖 _vocab！查询词可能不在任何已记忆的词汇表中。
        这让 TF 降级路径在冷启动时也能工作（记忆库为空时也能对查询做词袋匹配）。
        """
        words = text.lower().split()
        total = len(words)
        if total == 0:
            return {}
        counts = Counter(words)
        return {w: c / total for w, c in counts.items()}

    # ────────────────────────────────────────────────────────────────
    #  相似度计算（静态方法）
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        """
        稠密向量的余弦相似度。

        公式：cos(a,b) = (a·b) / (‖a‖ × ‖b‖)

        用于 embedding 语义相似度计算（主召回路径 + consolidate 去重比较）。

        预处理（快速失败）：
          • 任一为空 → 0.0（没有 embedding 无法比较）
          • 维度不等 → 0.0（不同模型生成的 embedding 不能比较，如 2560维 vs 3072维）

        复杂度：O(d)，其中 d = 向量维度（2560），对所有候选计算时开销可控。

        结果范围：[-1, 1]，但实际上 embedding 向量通常非负，输出在 [0, 1] 范围。
        """
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _cosine_tf(a: Dict[str, float], b: Dict[str, float]) -> float:
        """
        稀疏 TF 向量的余弦相似度。

        与 _cosine() 的区别：
          _cosine()    — 稠密向量（2560 维 float list），每个维度都有值
          _cosine_tf() — 稀疏向量（Dict[str, float]），只有非零项参与计算

        计算方式：
          dot = Σ a[k] × b[k]，其中 k 是 a 和 b 的 key 并集（不在某方 dict 中的默认 0）
          norm = √Σ v²，只对有值的 key 求平方和

        Args:
            a: 查询文本的 TF 向量（由 _text_to_tf 生成）
            b: Item 的 TF 向量（由 _build_tf_for 生成）

        复杂度：O(|keys|)，其中 |keys| 是两个 Dict key 集合的大小。
        通常远小于 _cosine() 的 2560 维。
        """
        if not a or not b:
            return 0.0
        all_keys = set(a.keys()) | set(b.keys())
        dot = sum(a.get(k, 0) * b.get(k, 0) for k in all_keys)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
