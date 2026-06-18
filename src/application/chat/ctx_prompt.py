"""Schema-driven prompt 上下文装配的内部封装"""

from src.domain.promptctx.source import Query
from src.domain.promptctx.assembler import ContextAssembler, SourceRegistry
from src.domain.promptctx.schema import default_schemas
from src.domain.promptctx.source_taskmem import TaskMemBuffer, StepObservation
from src.domain.promptctx.source_tools import ToolStateTracker, ToolCallTrace


class PromptCtx:
    def __init__(self):
        self.assembler: ContextAssembler = None
        self.task_mem: TaskMemBuffer = TaskMemBuffer(20)
        self.tool_tracker: ToolStateTracker = ToolStateTracker(10)

    def assemble(self, q: Query) -> str:
        if self.assembler is None:
            return ""
        rc = self.assembler.assemble(q)
        return rc.render()

    def reset_task_mem(self):
        if self.task_mem:
            self.task_mem.reset()

    def push_task_mem(self, obs: StepObservation):
        if self.task_mem:
            self.task_mem.push(obs)

    def record_tool_call(self, trace: ToolCallTrace):
        if self.tool_tracker:
            self.tool_tracker.record(trace)
