from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

SECTIONS = ["hosts", "ports", "findings", "credentials", "tasks", "reflections"]

@dataclass
class BlackboardEntry:
    section: str
    key: str
    value: Any
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    version: int = 1
