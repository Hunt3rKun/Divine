from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List

from llm_client import TokenUsage


class PlanningMode(Enum):
    """规划模式枚举"""
    INITIAL = "initial"           # 初始规划：从零开始创建任务图
    DYNAMIC = "dynamic"           # 动态规划：基于执行反馈调整计划
    BRANCH_REPLAN = "branch_replan"  # 分支再生：为失败分支生成替代方案




class GraphOperationCommand(Enum):
    """图操作命令枚举"""
    ADD_NODE = "ADD_NODE"         # 添加节点
    UPDATE_NODE = "UPDATE_NODE"   # 更新节点
    REMOVE_NODE = "REMOVE_NODE"   # 移除节点
    ADD_EDGE = "ADD_EDGE"         # 添加边
    REMOVE_EDGE = "REMOVE_EDGE"   # 移除边


@dataclass
class GraphOperation:
    """图操作数据结构"""
    command: GraphOperationCommand
    node_id: Optional[str] = None
    node_data: Optional[Dict[str, Any]] = None
    updates: Optional[Dict[str, Any]] = None
    edge_data: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphOperation":
        """从字典创建图操作"""
        command = GraphOperationCommand(data.get("command", "ADD_NODE"))
        return cls(
            command=command,
            node_id=data.get("node_id"),
            node_data=data.get("node_data"),
            updates=data.get("updates"),
            edge_data=data.get("edge_data"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result: Dict[str, Any] = {"command": self.command.value}
        if self.node_id:
            result["node_id"] = self.node_id
        if self.node_data:
            result["node_data"] = self.node_data
        if self.updates:
            result["updates"] = self.updates
        if self.edge_data:
            result["edge_data"] = self.edge_data
        return result




@dataclass
class PlanningThought:
    """规划思考过程"""
    step1_analysis: str = ""           # 信息提取与分析
    step2_decomposition: str = ""      # 任务拆分
    step3_dependency_analysis: str = ""  # 依赖建模
    step4_summary: str = ""            # 计划摘要

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanningThought":
        """从字典创建"""
        return cls(
            step1_analysis=data.get("step1_analysis", ""),
            step2_decomposition=data.get("step2_decomposition", ""),
            step3_dependency_analysis=data.get("step3_dependency_analysis", ""),
            step4_summary=data.get("step4_summary", ""),
        )



@dataclass
class PlanningResult:
    """规划结果"""
    thought: PlanningThought
    graph_operations: List[GraphOperation]
    global_mission_accomplished: bool = False
    raw_response: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], raw_response: str = None) -> "PlanningResult":
        """从字典创建"""
        thought = PlanningThought.from_dict(data.get("thought", {}))
        operations = [
            GraphOperation.from_dict(op)
            for op in data.get("graph_operations", [])
        ]
        return cls(
            thought=thought,
            graph_operations=operations,
            global_mission_accomplished=data.get("global_mission_accomplished", False),

            raw_response=raw_response,
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "thought": {
                "step1_analysis": self.thought.step1_analysis,
                "step2_decomposition": self.thought.step2_decomposition,
                "step3_dependency_analysis": self.thought.step3_dependency_analysis,
                "step4_summary": self.thought.step4_summary,
            },
            "graph_operations": [op.to_dict() for op in self.graph_operations],
            "global_mission_accomplished": self.global_mission_accomplished,
        }



@dataclass
class Planning:
    """单次规划记录"""
    timestamp: float
    mode: PlanningMode
    strategy: str
    outcome_summary: str







@dataclass
class PlannerContext:
    """规划器上下文"""

    # 规划模式
    planning_mode: PlanningMode = PlanningMode.INITIAL

    # 目标任务
    final_goal: str = ""

    # 已生成的所有图操作
    all_operations: List[GraphOperation] = field(default_factory=list)

    # 任务完成标志
    mission_completed: bool = False

    # 规划历史
    planning_history: List[Planning] = field(default_factory=list)

    total_usage: TokenUsage = field(default_factory=TokenUsage)  # 累计token使用
    total_cost: float = 0.0  # 累计费用

    # 检索到的经验
    retrieved_experience: str = ""

    _needs_compression: bool = False  # 是否需要压缩标志


    def add_operation(self, operations: List[GraphOperation]):
        """添加图操作到上下文"""
        self.all_operations.extend(operations)


    def get_planning_history(self,count: int=3) -> List[Planning]:
        """获取规划历史"""
        return self.planning_history[-count:]


    def add_planning_record(self, planning: Planning):
        """添加规划记录"""
        self.planning_history.append(planning)
