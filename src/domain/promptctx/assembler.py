"""ContextAssembler：按 Mode 选 Schema，并发调各 source 填充槽位"""

import asyncio
import threading
from typing import Dict, List, Optional

from src.domain.promptctx.source import ContextSource, Query
from src.domain.promptctx.slot import Slot, SlotKind, FilledSlot, ContextItem
from src.domain.promptctx.schema import (
    RuntimeContextSchema, default_schemas, DEFAULT_GLOBAL_TOKEN_BUDGET, slot_priority
)
from src.domain.promptctx.context import RuntimeContext


class SourceRegistry:
    """按 SlotKind 分组注册 ContextSource"""

    def __init__(self):
        self._mu = threading.RLock()
        self._sources: Dict[SlotKind, List[ContextSource]] = {}

    def register(self, source: ContextSource):
        with self._mu:
            all_kinds = [
                SlotKind.PROFILE, SlotKind.PLANNER, SlotKind.TASK_MEMORY,
                SlotKind.TOOL_STATE, SlotKind.CONSTRAINTS, SlotKind.RECALL_MEMORY,
            ]
            for kind in all_kinds:
                if source.supports(kind):
                    self._sources.setdefault(kind, []).append(source)

    def for_kind(self, kind: SlotKind) -> List[ContextSource]:
        with self._mu:
            return list(self._sources.get(kind, []))


class ContextAssembler:
    """装配入口：根据 Mode 选 Schema，并发调各 source 填充槽位"""

    def __init__(self, schemas: Optional[Dict[str, RuntimeContextSchema]] = None,
                 registry: Optional[SourceRegistry] = None,
                 global_limit: int = DEFAULT_GLOBAL_TOKEN_BUDGET):
        self._schemas = schemas or default_schemas()
        self._registry = registry or SourceRegistry()
        self._global_limit = global_limit

    def assemble(self, q: Query) -> RuntimeContext:
        schema = self._schemas.get(q.mode, self._schemas.get("chat", RuntimeContextSchema()))

        filled: Dict[int, FilledSlot] = {}

        # 使用线程池并发调各槽位对应的 source
        threads = []
        results_lock = threading.Lock()

        def fill(idx: int, slot: Slot):
            fs = self._fill_slot(slot, q)
            with results_lock:
                filled[idx] = fs

        for idx, slot in enumerate(schema.slots):
            t = threading.Thread(target=fill, args=(idx, slot))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        rc = RuntimeContext(
            schema=schema,
            filled=[filled.get(i, FilledSlot(kind=schema.slots[i].kind, skipped=True,
                                              reason="no source"))
                    for i in range(len(schema.slots))],
        )

        self._apply_global_budget(rc)
        return rc

    def _fill_slot(self, slot: Slot, q: Query) -> FilledSlot:
        sources = self._registry.for_kind(slot.kind)
        if not sources:
            return FilledSlot(kind=slot.kind, skipped=slot.required,
                              reason="no source registered")

        all_items: List[ContextItem] = []
        for src in sources:
            try:
                items = src.fetch(slot, q)
                all_items.extend(items)
            except Exception:
                pass

        if not all_items:
            return FilledSlot(kind=slot.kind, skipped=(not slot.required),
                              reason="source returned empty")

        # 单槽位 token budget 裁剪
        if slot.filter.token_budget > 0:
            all_items = _trim_by_budget(all_items, slot.filter.token_budget)

        return FilledSlot(kind=slot.kind, items=all_items)

    def _apply_global_budget(self, rc: RuntimeContext):
        total = sum(len(item.text) for fs in rc.filled for item in fs.items)
        if total <= self._global_limit:
            return

        # 按优先级从低到高排序逐步裁剪
        order = list(range(len(rc.filled)))
        order.sort(key=lambda i: slot_priority(rc.filled[i].kind), reverse=True)

        for idx in order:
            if total <= self._global_limit:
                break
            fs = rc.filled[idx]
            while fs.items and total > self._global_limit:
                last = fs.items[-1]
                total -= len(last.text)
                fs.items = fs.items[:-1]
            if not fs.items:
                fs.skipped = (not rc.schema.slots[idx].required)
                fs.reason = "global budget exceeded"


def _trim_by_budget(items: List[ContextItem], budget: int) -> List[ContextItem]:
    total = 0
    for i, item in enumerate(items):
        total += len(item.text)
        if total > budget:
            return items[:i]
    return items
