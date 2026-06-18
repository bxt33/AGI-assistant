"""用户偏好记忆：支持 LLM NER 提取 + 规则兜底。

=============================================================================
                    🧠 记忆系统 · 偏好维度：用户画像
=============================================================================

四维记忆中的独立维度：
  ShortTerm  — "刚才说了什么"      （对话历史，滑动窗口，不持久化）
  LongTerm   — "以前聊过什么"      （语义召回 + TF 降级，三阶段合并淘汰，持久化PG）
  GraphMemory— "哪些记忆有关联"    （Neo4j FOLLOWS/SIMILAR_TO 图扩展）
  Preference — "用户是谁 / 喜欢什么"（KV 画像）← 这里

=============================================================================
                        📦 数据结构：最简单的 Dict[str, str]
=============================================================================

Preference 内部就是一个 Dict[str, str]，例如：

  {
    "姓名": "张三",
    "职业": "后端开发工程师",
    "回答风格": "简洁",
    "常用语言": "Python",
    "位置": "北京",
    "偏好图表类型": "折线图",
    ...
  }

为什么不用 LTM 的 Item 结构（带 embedding + cosine 召回）？
  偏好是精确的键值对，操作是"查姓名是什么"而非"找与姓名相似的偏好"。
  ① 不需要语义向量化——"姓名"就是"姓名"，不需要余弦相似度匹配
  ② 不需要去重——同一 key 的新值直接覆盖旧值
  ③ 不需要三阶段合并淘汰——偏好数量极小（通常 < 50 项）
  用 Dict 比 List[Item] 节省 100 倍代码复杂度。

=============================================================================
                        🔌 两种提取方式
=============================================================================

方式 1（主路径）：LLM NER 提取
  在 core_agent._extract_and_save_prefs() 中实现：
    调用 LLMClient.extract_preferences(用户消息) → 返回 {key: value, ...}
    → Preference.save_batch(kvs) → 写入内存 Dict
    → 同时持久化到 PG preferences 表（如果 repo 可用）

  示例：用户说"我是张三，在北京做后端，喜欢简洁的回答"
    → LLM 提取：{"姓名":"张三", "位置":"北京", "职业":"后端开发", "回答风格":"简洁"}

方式 2（降级路径）：正则规则兜底 — extract_rule_based()
  当 LLM 不可用时，用简单正则匹配常见中文句式：
    "我喜欢xxx"  → {"喜好": "xxx"}
    "我叫xxx"    → {"姓名": "xxx"}
    "我在xxx"    → {"位置": "xxx"}
  只覆盖 3 种模式，效果远不如 LLM，但保证最坏情况下仍有基础功能。

=============================================================================
                    🖨️ 在提示词中的呈现方式
=============================================================================

ProfileSource（promptctx/source_profile.py）在每次 assemble() 时调用：
  pref.load() → 获取全部偏好 KV → 格式化为 ContextItem 列表 → 填入 PROFILE 槽位

最终渲染为：
  【用户画像】
  - 姓名: 张三
  - 回答风格: 简洁
  - 常用语言: Python

所有 4 种 Schema（CHAT/TOOL/REACT/RAG）都包含 PROFILE 槽位，
因此用户画像在所有模式下的提示词中都会出现——让 LLM 始终"认识"用户。

=============================================================================
                    🏗️ 与持久化的关系
=============================================================================

当前实现：纯内存 Dict，没有内置 PG 持久化。
意味着进程重启后偏好丢失——需要在首次对话中重新建立。

为什么当前不持久化？
  ① 偏好数据极小（< 50 条），重建成本低（一次 LLM 调用即可提取）
  ② core_agent 层可以在 save_batch 后额外调用 PG repo——持久化逻辑上移
  ③ Preference 类保持纯粹——只关注"存储和提取"，不关心"存到哪里"

未来优化方向：在 __init__ 中注入 pg_repo 参数，save/save_batch 时自动持久化。
=============================================================================
"""

import threading
import re
from typing import Dict


class Preference:
    """
    用户偏好存储：内存 Dict + 持久化回调（预留）。

    ── 数据结构 ──
    self._data: Dict[str, str]
      键值对存储。key 是偏好项名称（如"姓名"、"回答风格"），
      value 是偏好值（如"张三"、"简洁"）。

    ── 核心操作 ──
    save(key, value)     — 覆盖式写入单条偏好
    save_batch(kvs)      — 批量合并写入（LLM 一次性提取多条时使用）
    load(user_id)        — 加载全部偏好（user_id 预留用于多用户，当前忽略）
    snapshot()           — 返回偏好副本（与 load 功能相同，别名）
    extract_rule_based() — 静态方法：正则兜底提取（无 LLM 时的降级路径）

    ── 线程安全 ──
    threading.RLock() 保护所有读写：
      • save / save_batch — core_agent 后处理线程调用
      • load / snapshot   — promptctx assemble 线程调用
      两者可能并发（一边写入新偏好，一边组装提示词），需要锁保护。

    ── 为什么不用 config 文件存储偏好？ ──
    config.yaml 是静态配置（API key、数据库地址等），不应频繁写入。
    偏好是动态数据——每次对话都可能更新——适合内存 Dict + 可选 PG 持久化。
    """

    def __init__(self):
        self._mu = threading.RLock()
        self._data: Dict[str, str] = {}   # {偏好键: 偏好值}

    def save(self, key: str, value: str):
        """
        覆盖式写入单条偏好。

        调用时机：
          core_agent._extract_and_save_prefs() 逐条保存时使用。
          或用户直接说"叫我张三" → 立即保存 {"姓名": "张三"}。

        覆盖语义：如果 key 已存在，新值直接覆盖旧值。
        这是正确的行为——偏好本质上是最新状态，不需要保留历史。

        示例：
          pref.save("回答风格", "简洁")
          pref.save("回答风格", "详细")  → "简洁"被覆盖为"详细"

        Args:
            key:   偏好键名，如 "姓名"、"回答风格"、"常用语言"
            value: 偏好值，如 "张三"、"简洁"、"Python"
        """
        with self._mu:
            self._data[key] = value

    def save_batch(self, kvs: Dict[str, str]):
        """
        批量合并写入偏好。

        调用时机：
          LLM 一次性从对话中提取多条偏好后调用。
          例如 LLM 返回 {"姓名":"张三", "回答风格":"简洁", "常用语言":"Python"}
          → save_batch() 一次性写入。

        与多次 save() 的区别：
          save_batch() 一次加锁、一次更新，减少锁竞争。
          当多条偏好来自同一轮 LLM 解析时，原子写入更合理。

        Args:
            kvs: 偏好字典，{"key": "value", ...}
        """
        with self._mu:
            self._data.update(kvs)

    def load(self, user_id: str = "default") -> Dict[str, str]:
        """
        加载全部偏好。

        user_id 参数当前忽略（单用户模式），预留用于多用户场景（未来扩展）。

        调用时机：
          ProfileSource.fetch() → pref.load() → 转为 ContextItem → 注入提示词
          前端 API 获取用户画像

        Returns:
          偏好的浅拷贝（dict 副本），防止外部代码意外修改内部数据。
        """
        with self._mu:
            return dict(self._data)

    def snapshot(self) -> Dict[str, str]:
        """
        返回偏好快照副本。

        与 load() 功能相同，语义化别名。
        调用时机：前端展示、日志输出、PromptCtx 读取。

        Returns:
           dict 副本（线程安全）
        """
        with self._mu:
            return dict(self._data)

    @staticmethod
    def extract_rule_based(msg: str) -> Dict[str, str]:
        """
        规则兜底：无 LLM 时用正则表达式从文本中提取偏好。

        ⚠️ 这是降级路径（fallback），精度远不如 LLM 提取。只覆盖最常见的 3 种中文句式。

        ── 支持的模式 ──
        ① "我喜欢xxx"  → {"喜好": "xxx"}
          如 "我喜欢喝咖啡" → {"喜好": "喝咖啡"}
        ② "我叫xxx"    → {"姓名": "xxx"}
          如 "我叫张三" → {"姓名": "张三"}
        ③ "我在xxx"    → {"位置": "xxx"}
          如 "我在北京" → {"位置": "北京"}

        ── 匹配策略 ──
        正则 (.+?) 做非贪婪匹配，到第一个分隔符（逗号、句号、感叹号等）或文本末尾停止。
        这能避免 "我喜欢咖啡，也喜欢茶" 匹配成 "咖啡，也喜欢茶"。

        ── 局限性 ──
        ① 只支持中文常见句式（"I like X" 不匹配）
        ② 只覆盖 3 种固定模式，无法提取复杂偏好（如"我更倾向于图表而非文字描述"）
        ③ 无法处理否定句式（"我不太喜欢Python" → 会提取 "不太喜欢Python"，而非否定）
        ④ 中文分词粗糙——基于标点符号切句，不是语义切句

        ── 主路径 vs 降级路径 ──
        LLM 可用时：优先使用 LLMClient.extract_preferences()，可提取任意形式的偏好。
        LLM 不可用时：回退到本方法，覆盖最基础的 3 种模式。

        Args:
            msg: 用户原始消息文本

        Returns:
            提取到的偏好字典，可能为空 {}
        """
        result = {}

        # "我喜欢xxx" — 匹配到第一个中文标点或空白符或文本结束
        m = re.search(r'我喜欢(.+?)(?:[，。！？\s]|$)', msg)
        if m:
            result["喜好"] = m.group(1).strip()

        # "我叫xxx" — 提取姓名
        m = re.search(r'我叫(.+?)(?:[，。！？\s]|$)', msg)
        if m:
            result["姓名"] = m.group(1).strip()

        # "我在xxx" — 提取位置信息
        m = re.search(r'我在(.+?)(?:[，。！？\s]|$)', msg)
        if m:
            result["位置"] = m.group(1).strip()

        return result
