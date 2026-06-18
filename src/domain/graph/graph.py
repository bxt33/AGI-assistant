"""任务图 (DAG) 领域模型：将 ReAct 从串行链路升级为可并行、可竞速的图结构调度"""

from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque


class NodeType(str, Enum):
    TOOL = "tool"
    THINK = "think"
    AGGREGATE = "aggregate"


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class Node:
    id: str = ""
    type: NodeType = NodeType.TOOL
    name: str = ""
    tool_name: str = ""
    params: Dict[str, str] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    race_group: str = ""
    status: NodeStatus = NodeStatus.PENDING
    result: str = ""
    error: str = ""
    retry_count: int = 0


class TaskGraph:
    """有向无环图，通过拓扑排序决定节点执行顺序"""

    def __init__(self, nodes: List[Node]):
        self.nodes: Dict[str, Node] = {}
        self.adj_list: Dict[str, List[str]] = {}
        self.in_degree: Dict[str, int] = {}
        self._levels: Optional[List[List[str]]] = None

        # 注册所有节点
        for n in nodes:
            n.status = NodeStatus.PENDING
            self.nodes[n.id] = n
            self.adj_list[n.id] = []
            self.in_degree[n.id] = 0

        # 建立邻接表和入度
        for n in nodes:
            for dep in n.depends_on:
                if dep in self.nodes:
                    self.adj_list[dep].append(n.id)
                    self.in_degree[n.id] += 1

    def topological_levels(self) -> List[List[str]]:
        """Kahn 算法按层拓扑排序，同层节点可并行"""
        if self._levels is not None:
            return self._levels

        in_deg = dict(self.in_degree)
        levels = []
        processed = 0

        while True:
            ready = [nid for nid, d in in_deg.items() if d == 0]
            if not ready:
                break

            levels.append(ready)
            processed += len(ready)

            for nid in ready:
                in_deg[nid] = -1
                for downstream in self.adj_list.get(nid, []):
                    in_deg[downstream] -= 1

        if processed != len(self.nodes):
            raise ValueError(f"Graph has cycle: processed {processed}/{len(self.nodes)} nodes")

        self._levels = levels
        return levels

    def ready_nodes(self) -> List[str]:
        """返回当前入度为 0 的待执行节点"""
        ready = []
        for nid, d in self.in_degree.items():
            if d == 0 and self.nodes[nid].status == NodeStatus.PENDING:
                ready.append(nid)
        return ready

    def mark_done(self, nid: str) -> List[str]:
        """标记节点完成，更新下游入度，返回新就绪的节点"""
        self.in_degree[nid] = -1
        newly_ready = []
        for downstream in self.adj_list.get(nid, []):
            self.in_degree[downstream] -= 1
            if self.in_degree[downstream] == 0 and self.nodes[downstream].status == NodeStatus.PENDING:
                newly_ready.append(downstream)
        return newly_ready

    def race_groups(self) -> Dict[str, List[str]]:
        """按 race_group 对节点分组"""
        groups: Dict[str, List[str]] = {}
        for nid, n in self.nodes.items():
            if n.race_group:
                groups.setdefault(n.race_group, []).append(nid)
        return groups

    def validate(self) -> Optional[str]:
        """图的合法性检测"""
        for n in self.nodes.values():
            for dep in n.depends_on:
                if dep not in self.nodes:
                    return f"Node {n.id} depends on nonexistent node {dep}"
        try:
            self.topological_levels()
            return None
        except ValueError as e:
            return str(e)

    def set_node_status(self, nid: str, status: NodeStatus):
        if nid in self.nodes:
            self.nodes[nid].status = status

    def set_node_result(self, nid: str, result: str):
        if nid in self.nodes:
            self.nodes[nid].result = result

    def set_node_error(self, nid: str, error: str):
        if nid in self.nodes:
            self.nodes[nid].error = error

    def successful_results(self) -> List[str]:
        """返回所有成功节点的结果列表"""
        results = []
        for n in self.nodes.values():
            if n.status == NodeStatus.DONE and n.result:
                results.append(f"[{n.tool_name}] {n.result}")
        return results

    def summary(self) -> str:
        try:
            levels = self.topological_levels()
        except ValueError as e:
            return f"graph invalid: {e}"

        lines = [f"graph: {len(self.nodes)} nodes, {len(levels)} levels"]
        for i, level in enumerate(levels):
            names = [f"{nid}({self.nodes[nid].tool_name})" for nid in level]
            lines.append(f"  L{i}: {', '.join(names)}")
        return "\n".join(lines)
