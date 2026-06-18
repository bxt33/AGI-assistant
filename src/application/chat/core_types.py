"""UnifiedAgent 用到的数据结构（请求 / 响应 / 任务 / SSE 事件）"""

from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


class StepType(str, Enum):
    THOUGHT = "Thought"
    ACTION = "Action"
    OBSERVATION = "Observation"
    FINAL_ANSWER = "Final Answer"


@dataclass
class ReActStep:
    type: StepType
    content: str = ""
    tool: str = ""
    params: Dict[str, str] = field(default_factory=dict)


class TaskStepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass
class TaskStep:
    id: int = 0
    name: str = ""
    tool_name: str = ""
    params: Dict[str, str] = field(default_factory=dict)
    status: TaskStepStatus = TaskStepStatus.PENDING
    result: str = ""
    error: str = ""
    retry_count: int = 0


@dataclass
class TaskState:
    task_id: str = ""
    query: str = ""
    status: str = ""
    phase: str = ""
    steps: List[TaskStep] = field(default_factory=list)
    current_step: int = 0
    interrupted_at: int = 0
    result: str = ""
    graph: Any = None


@dataclass
class Snapshot:
    state: TaskState
    timestamp: str = ""


@dataclass
class Response:
    query: str = ""
    answer: str = ""
    mode: str = ""
    steps: List[ReActStep] = field(default_factory=list)
    tool_call: Any = None
    search_results: List = field(default_factory=list)
    task: Optional[TaskState] = None
    extracted_info: str = ""
    short_term_count: int = 0
    long_term_count: int = 0
    preferences: Dict[str, str] = field(default_factory=dict)
    interrupted: bool = False


@dataclass
class StreamEvent:
    type: str = ""
    data: Any = None


@dataclass
class ChatOptions:
    use_rag: bool = False
    selected_tools: Optional[List[str]] = None
    explicit: bool = False
