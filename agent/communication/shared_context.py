import threading
import time
from enum import Enum
from typing import Optional, Dict, List

import networkx as nx

from logger_config import log


class SubtaskStatus(Enum):
    """子任务状态枚举"""
    PENDING = "pending"                 # 待执行
    IN_PROGRESS = "in_progress"         # 执行中
    COMPLETED = "completed"             # 已完成
    FAILED = "failed"                   # 失败
    DEPRECATED = "deprecated"           # 已废弃 (被规划器移除)
    STALLED_ORPHAN = "stalled_orphan"   # 孤儿节点 (依赖被移除)
    COMPLETED_ERROR = "completed_error" # 完成但有错误




class NodeType(Enum):
    """节点类型枚举"""
    TASK = "seed_task"                      # 根任务节点 (对应主目标)
    SUBTASK = "subtask"                # 子任务节点 (Planner 生成)
    EXECUTION_STEP = "micro_execution_task"  # 执行步骤 (Executor 产生)



class EdgeType(Enum):
    """边类型枚举"""
    DECOMPOSITION = "decomposition"  # 任务分解: task → subtask
    DEPENDENCY = "dependency"        # 依赖关系: subtask → subtask
    EXECUTION = "execution"          # 执行链: subtask → micro_execution_task





class SharedContext:
    def __init__(self):
        self.graph_manager = None
        self._intelligence = None


