from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


AuditStatus = Literal["success", "failed", "partial", "uncertain", "blocked"]
FailureLevel = Literal[
    "none",
    "execution_failure",
    "cognitive_failure",
    "strategic_failure",
    "constraint_failure",
    "insufficient_evidence",
]
PlanningStrategy = Literal["expand", "regenerate_node", "replan_branch"]


@dataclass
class TaskJudgement:
    status: AuditStatus = "uncertain"
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
    task_id: str
    task_judgement: TaskJudgement
    audit_result: AuditResult = field(default_factory=AuditResult)
    failure_attribution: FailureAttribution = field(default_factory=FailureAttribution)
    planning_feedback: PlanningFeedback = field(default_factory=PlanningFeedback)

