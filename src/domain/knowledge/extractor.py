"""实体关系抽取：通过 LLM 从文本中抽取实体和关系"""

import json
import logging
from typing import Callable, Optional

from src.domain.knowledge.types import Entity, Relation, ExtractResult, EntityType

logger = logging.getLogger(__name__)

_EXTRACT_SYSTEM_PROMPT = """你是一个信息抽取专家。从给定文本中抽取命名实体和实体间关系。

实体类型（type 字段只能用以下值）：
- Person（人物）
- Organization（组织/公司/机构）
- Location（地点/地区）
- Concept（概念/技术/思想）
- Event（事件）
- Product（产品/工具）
- Unknown（其他）

关系类型（rel_type 字段只能用以下值）：
- RELATES_TO（相关）
- PART_OF（属于/是...的一部分）
- CAUSES（导致/引发）
- DESCRIBES（描述/介绍）
- MENTIONS（提及）
- WORKS_FOR（工作于）
- LOCATED_IN（位于）

输出格式（只输出 JSON，不加任何说明）：
{
  "entities": [{"name":"实体名","type":"类型"}],
  "relations": [{"from":"实体A","to":"实体B","rel_type":"关系类型"}]
}

如果文本中没有可抽取的实体，输出 {"entities":[],"relations":[]}"""

_VALID_ENTITY_TYPES = {e.value for e in EntityType}
_VALID_REL_TYPES = {"RELATES_TO", "PART_OF", "CAUSES", "DESCRIBES",
                    "MENTIONS", "WORKS_FOR", "LOCATED_IN"}


class Extractor:
    """通过注入的 LLM 回调从文本中抽取实体和关系"""

    def __init__(self, llm_fn: Optional[Callable[[str, str], str]] = None):
        self._llm_fn = llm_fn

    def extract(self, text: str) -> ExtractResult:
        if not self._llm_fn or not text.strip():
            return ExtractResult()

        try:
            raw = self._llm_fn(_EXTRACT_SYSTEM_PROMPT, f"文本：\n{text}")
            raw = raw.strip()
            raw = raw.removeprefix("```json").removeprefix("```")
            raw = raw.removesuffix("```").strip()

            data = json.loads(raw)
            result = ExtractResult(
                entities=[
                    Entity(name=e["name"].strip(), type=EntityType(e.get("type", "Unknown")))
                    for e in data.get("entities", [])
                    if e.get("name", "").strip()
                ],
                relations=[
                    Relation(
                        from_name=r.get("from", "").strip(),
                        to_name=r.get("to", "").strip(),
                        rel_type=r.get("rel_type", "RELATES_TO"),
                    )
                    for r in data.get("relations", [])
                    if r.get("from", "").strip() and r.get("to", "").strip()
                ],
            )
            return result
        except Exception as e:
            logger.warning(f"实体关系抽取失败: {e}")
            return ExtractResult()
