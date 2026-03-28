from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import uuid4


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingType(str, Enum):
    ASSET = "asset"
    CREDENTIAL = "credential"
    VULNERABILITY = "vulnerability"
    KNOWLEDGE = "knowledge"


@dataclass
class Finding:
    id: str = field(default_factory=lambda: uuid4().hex[:8])
    type: FindingType = FindingType.KNOWLEDGE
    severity: Severity = Severity.INFO
    title: str = ""
    detail: dict = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    source_task: str = ""
    discovered_at: datetime = field(default_factory=datetime.now)
