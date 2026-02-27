import threading
from datetime import time
from typing import List, Optional, Dict, Any

import networkx as nx

from agent.communication.shared_context import NodeType, SubtaskStatus, EdgeType
from logger_config import log


class GraphManager:
    def __init__(
            self,
            task_id: str,
            goal: str,
    ):
        self.task_id = task_id
        self.goal = goal
        self.graph = nx.DiGraph()

        self._lock = threading.RLock()
        self._initialize_graph(goal)

    def _initialize_graph(self,goal: str):
        """初始化图，添加代表整体任务的根节点"""
        node_data = {
            "type": NodeType.TASK.value,
            "goal": goal,
            "status": SubtaskStatus.IN_PROGRESS.value
        }
        self.graph.add_node(self.task_id, **node_data)

    def add_subtask_node(
        self,
        subtask_id: str,
        description: str,
        dependencies: List[str],
        priority: int = 1,
        reason: str = "",
        completion_criteria: str = "",
        # required_capability: str = "generic",
        mission_briefing: Optional[Dict] = None
    ) -> str:
        """
        添加子任务节点

        Args:
            subtask_id: 子任务 ID
            description: 任务描述
            dependencies: 依赖的任务 ID 列表
            priority: 优先级 (数值越小优先级越高)
            reason: 创建原因
            completion_criteria: 完成条件
            # required_capability: 所需能力
            mission_briefing: 任务简报

        Returns:
            子任务 ID

        Raises:
            DuplicateNodeError: 节点已存在
        """
        with self._lock:
            if self.graph.has_node(subtask_id):
                log.warning(f"[GraphManager] 子任务节点已存在，跳过: {subtask_id}")
                return subtask_id

            now = time.time()
            node_data = {
                "type": NodeType.SUBTASK.value,
                "description": description,
                "status": SubtaskStatus.PENDING.value,
                "priority": priority,
                "reason": reason,
                "completion_criteria": completion_criteria,
                # "required_capability": required_capability,
                "mission_briefing": mission_briefing,
                "reflection": None,
                "summary": None,
                "artifacts": [],
                "proposed_changes": [],
                "conversation_history": [],
                "turn_counter": 0,
                "execution_summary_cache": None,
                "execution_summary_last_sequence": 0,
                "execution_summary_updated_at": None,
                "created_at": now,
                "updated_at": now,
            }

            self.graph.add_node(subtask_id, **node_data)


            # 创建边
            if not dependencies:
                # 无依赖则连接到根节点
                self.graph.add_edge(
                    self.task_id,
                    subtask_id,
                    type=EdgeType.DECOMPOSITION.value
                )

            else:
                # 创建依赖边
                for dep_id in dependencies:
                    if self.graph.has_node(dep_id):
                        self.graph.add_edge(
                            dep_id,
                            subtask_id,
                            type=EdgeType.DEPENDENCY.value
                        )

                    else:
                        log.warning(f"[GraphManager] 依赖节点不存在: {dep_id}")

            log.debug(f"[GraphManager] 子任务已添加: {subtask_id} | deps={dependencies}")
            return subtask_id

    def get_failed_nodes(self) -> Dict[str, Any]:
        """获取失败节点"""
        failed = {}
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == NodeType.SUBTASK.value:
                status = data.get("status")
                if status in (
                        SubtaskStatus.FAILED.value,
                        SubtaskStatus.STALLED_ORPHAN.value,
                        SubtaskStatus.COMPLETED_ERROR.value
                ):
                    failed[node_id] = data
        return failed


    def get_descendants(self, node_id: str) -> set:
        """
        在图结构中查找指定节点的所有后代节点（即所有从该节点出发，通过有向边可以到达的下游节点，包括子节点、孙节点等）
        :param node_id:
        :return:
        """
        if not self.graph.has_node(node_id):
            return set()
        return nx.descendants(self.graph, node_id)



