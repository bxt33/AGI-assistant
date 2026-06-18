"""任务运行时：per-request 共享状态（当前任务、快照、取消函数）"""

import threading
from typing import Optional, List, Callable

from src.application.chat.core_types import TaskState, Snapshot


class TaskRuntime:
    def __init__(self):
        self._mu = threading.Lock()
        self._current_task: Optional[TaskState] = None
        self._snapshots: List[Snapshot] = []
        self._cancel_funcs: List[Callable[[], None]] = []

    def current_task(self) -> Optional[TaskState]:
        with self._mu:
            return self._current_task

    def set_task(self, t: Optional[TaskState]):
        with self._mu:
            self._current_task = t

    def snapshot_list(self) -> List[Snapshot]:
        with self._mu:
            return list(self._snapshots)

    def append_snapshot(self, snap: Snapshot):
        with self._mu:
            self._snapshots.append(snap)

    def register_cancel(self, cancel: Callable[[], None]) -> Callable[[], None]:
        def unregister():
            with self._mu:
                if cancel in self._cancel_funcs:
                    self._cancel_funcs.remove(cancel)
        with self._mu:
            self._cancel_funcs.append(cancel)
        return unregister

    def cancel_all(self):
        with self._mu:
            funcs = list(self._cancel_funcs)
            self._cancel_funcs.clear()
        for fn in funcs:
            try:
                fn()
            except Exception:
                pass
