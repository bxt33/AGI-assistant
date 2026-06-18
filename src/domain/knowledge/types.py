"""知识图谱类型定义"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List


class EntityType(str, Enum):
    PERSON = "Person"
    ORGANIZATION = "Organization"
    LOCATION = "Location"
    CONCEPT = "Concept"
    EVENT = "Event"
    PRODUCT = "Product"
    UNKNOWN = "Unknown"


@dataclass
class Entity:
    name: str = ""
    type: EntityType = EntityType.UNKNOWN
    doc_hash: str = ""
    chunk_id: int = 0
    pg_id: int = 0


@dataclass
class Relation:
    from_name: str = ""
    to_name: str = ""
    rel_type: str = "RELATES_TO"
    weight: float = 1.0
    doc_hash: str = ""
    chunk_id: int = 0
    pg_id: int = 0


@dataclass
class GraphSearchResult:
    chunk_id: int = 0
    pg_id: int = 0
    score: float = 0.0
    entities: List[str] = field(default_factory=list)
    hop_path: List[str] = field(default_factory=list)


@dataclass
class ExtractResult:
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)


@dataclass
class ChunkRef:
    id: int = 0
    pg_id: int = 0
    content: str = ""
