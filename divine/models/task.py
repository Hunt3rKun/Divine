from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from divine.models.common import PentestPhase, ExecutorType


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskNode:
    id: str
    description: str
    phase: PentestPhase
    executor_type: ExecutorType
    dependencies: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
