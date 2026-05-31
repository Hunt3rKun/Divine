"""Shared blackboard and dynamic DAG data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Mapping


NodeStatus = Literal[
    "pending",
    "running",
    "success",
    "failed",
    "partial",
    "uncertain",
    "pruned",
    "blocked",
    "completed",
]

EdgeType = Literal["dependency", "hypothesis", "validation", "alternative"]
RiskLevel = Literal["low", "medium", "high"]
FailureLevel = Literal[
    "none",
    "execution_failure",
    "cognitive_failure",
    "strategic_failure",
    "constraint_failure",
    "insufficient_evidence",
]
PlanningStrategy = Literal[
    "expand",
    "regenerate_node",
    "replan_branch",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TaskContext:
    task_id: str
    goal: str
    target: str
    scope: list[str]
    created_at: str = field(default_factory=utc_now_iso)
    max_iterations: int = 20
    max_consecutive_failures: int = 5


@dataclass
class TaskNode:
    node_id: str
    task_type: str
    description: str
    status: NodeStatus = "pending"
    dependencies: list[str] = field(default_factory=list)
    edge_type: EdgeType = "dependency"
    risk_level: RiskLevel = "low"
    success_criteria: list[str] = field(default_factory=list)
    assigned_executor: str | None = None
    attempt_count: int = 0
    max_attempts: int = 2
    evidence_refs: list[str] = field(default_factory=list)
    created_by: str = "planner_agent"
    created_from: str | None = None
    target: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def can_execute_after(self, completed_node_ids: set[str]) -> bool:
        return self.status == "pending" and all(dep in completed_node_ids for dep in self.dependencies)


@dataclass(frozen=True)
class TaskEdge:
    from_node: str
    to_node: str
    edge_type: EdgeType = "dependency"
    reason: str = ""


@dataclass
class Fact:
    fact_id: str
    type: str
    target: str
    key: str
    value: Any
    confidence: float
    source: str
    evidence_refs: list[str] = field(default_factory=list)
    status: str = "confirmed"

    @classmethod
    def from_mapping(
        cls,
        fact_id: str,
        payload: Mapping[str, Any],
        *,
        source: str,
        evidence_refs: list[str] | None = None,
        status: str = "confirmed",
    ) -> "Fact":
        return cls(
            fact_id=fact_id,
            type=str(payload.get("type") or "unknown"),
            target=str(payload.get("target") or ""),
            key=str(payload.get("key") or payload.get("type") or "value"),
            value=payload.get("value"),
            confidence=float(payload.get("confidence") or 0.0),
            source=source,
            evidence_refs=list(payload.get("evidence_refs") or evidence_refs or []),
            status=status,
        )


@dataclass
class Artifact:
    artifact_id: str
    artifact_type: str
    source: str
    node_id: str
    content: Mapping[str, Any]
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class TaskJudgement:
    status: NodeStatus
    completion_score: float = 0.0
    confidence: float = 0.0


@dataclass
class AuditResult:
    confirmed_facts: list[Mapping[str, Any]] = field(default_factory=list)
    candidate_findings: list[Mapping[str, Any]] = field(default_factory=list)
    vulnerabilities: list[Mapping[str, Any]] = field(default_factory=list)
    credentials: list[Mapping[str, Any]] = field(default_factory=list)
    sessions: list[Mapping[str, Any]] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    state_updates: list[Mapping[str, Any]] = field(default_factory=list)


@dataclass
class FailureAttribution:
    level: FailureLevel = "none"
    primary_cause: str | None = None
    secondary_causes: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reason: str | None = None


@dataclass
class PlanningFeedback:
    recommended_strategy: PlanningStrategy | None = None
    next_focus: str | None = None
    invalidated_hypothesis: str | None = None
    should_terminate: bool = False


@dataclass
class AuditFeedback:
    feedback_id: str
    node_id: str
    task_judgement: TaskJudgement
    audit_result: AuditResult = field(default_factory=AuditResult)
    failure_attribution: FailureAttribution = field(default_factory=FailureAttribution)
    planning_feedback: PlanningFeedback = field(default_factory=PlanningFeedback)


@dataclass
class ExecutionResult:
    execution_id: str
    node_id: str
    executor: str
    status: str
    summary: str = ""
    actions: list[Mapping[str, Any]] = field(default_factory=list)
    tool_results: list[Mapping[str, Any]] = field(default_factory=list)
    candidate_facts: list[Mapping[str, Any]] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    raw_output_refs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass(frozen=True)
class Event:
    event_id: str
    event_type: str
    message: str
    created_at: str = field(default_factory=utc_now_iso)
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannerOperation:
    operation: str
    target_node_id: str | None
    reason: str


@dataclass
class PlannerResult:
    status: str
    strategy: str
    rationale: str
    operations: list[PlannerOperation] = field(default_factory=list)
    added_nodes: list[str] = field(default_factory=list)
    updated_nodes: list[str] = field(default_factory=list)
    pruned_nodes: list[str] = field(default_factory=list)
    should_terminate: bool = False
