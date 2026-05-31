import asyncio
import networkx as nx
from divine.models.task import TaskNode, TaskStatus
from divine.models.common import PentestPhase, ExecutorType


class TaskDAG:
    def __init__(self):
        self._graph = nx.DiGraph()
        self._lock = asyncio.Lock()

    async def add_task(self, task: TaskNode) -> None:
        """Add task node + dependency edges + cycle detection. Rollback if cycle found."""
        async with self._lock:
            self._graph.add_node(task.id, task=task)
            for dep_id in task.dependencies:
                if dep_id in self._graph:
                    self._graph.add_edge(dep_id, task.id)
            if not nx.is_directed_acyclic_graph(self._graph):
                self._graph.remove_node(task.id)
                raise ValueError(f"Adding task '{task.id}' would create a cycle")

    async def remove_task(self, task_id: str) -> None:
        async with self._lock:
            if task_id in self._graph:
                self._graph.remove_node(task_id)

    async def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: dict = None,
        error: str = None,
    ) -> None:
        async with self._lock:
            if task_id not in self._graph:
                return
            task = self._graph.nodes[task_id]["task"]
            task.status = status
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error

    def get_task(self, task_id: str) -> TaskNode | None:
        if task_id not in self._graph:
            return None
        return self._graph.nodes[task_id]["task"]

    def get_all_tasks(self) -> list[TaskNode]:
        return [self._graph.nodes[n]["task"] for n in self._graph.nodes]

    def get_ready_tasks(self) -> list[TaskNode]:
        """Return PENDING tasks whose all predecessors are COMPLETED, sorted by priority desc."""
        ready = []
        for node_id in self._graph.nodes:
            task = self._graph.nodes[node_id]["task"]
            if task.status != TaskStatus.PENDING:
                continue
            predecessors = list(self._graph.predecessors(node_id))
            all_deps_done = all(
                self._graph.nodes[p]["task"].status == TaskStatus.COMPLETED
                for p in predecessors
            )
            if all_deps_done:
                ready.append(task)
        ready.sort(key=lambda t: t.priority, reverse=True)
        return ready

    def get_failed_tasks(self) -> list[TaskNode]:
        return [
            self._graph.nodes[n]["task"]
            for n in self._graph.nodes
            if self._graph.nodes[n]["task"].status == TaskStatus.FAILED
        ]

    def get_descendants(self, task_id: str) -> set[str]:
        if task_id not in self._graph:
            return set()
        return nx.descendants(self._graph, task_id)

    async def propagate_failure(self, task_id: str) -> list[str]:
        async with self._lock:
            descendants = (
                nx.descendants(self._graph, task_id)
                if task_id in self._graph
                else set()
            )
            skipped = []
            for desc_id in descendants:
                task = self._graph.nodes[desc_id]["task"]
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.SKIPPED
                    skipped.append(desc_id)
            return skipped

    async def apply_operations(self, operations: list[dict]) -> None:
        for op in operations:
            cmd = op.get("command", "")
            # 兼容 add_task/add_node 两种写法
            if cmd in ("add_node", "add_task"):
                data = op.get("node_data") or op
                task = TaskNode(
                    id=data["id"],
                    description=data.get("description", ""),
                    phase=PentestPhase(data.get("phase", "recon")),
                    executor_type=ExecutorType(data.get("executor_type", "recon")),
                    dependencies=data.get("dependencies", []),
                    priority=data.get("priority", 0),
                )
                await self.add_task(task)
            elif cmd in ("remove_node", "remove_task"):
                await self.remove_task(op.get("node_id", ""))
            elif cmd in ("update_node", "update_task"):
                node_id = op["node_id"]
                updates = op.get("updates", {})
                async with self._lock:
                    if node_id in self._graph:
                        task = self._graph.nodes[node_id]["task"]
                        for key, value in updates.items():
                            if hasattr(task, key):
                                setattr(task, key, value)

    @property
    def is_finished(self) -> bool:
        terminal = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED}
        return all(
            self._graph.nodes[n]["task"].status in terminal for n in self._graph.nodes
        )

    @property
    def stats(self) -> dict:
        counts = {
            "total": 0,
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "skipped": 0,
        }
        for n in self._graph.nodes:
            status = self._graph.nodes[n]["task"].status.value
            counts["total"] += 1
            counts[status] = counts.get(status, 0) + 1
        return counts
