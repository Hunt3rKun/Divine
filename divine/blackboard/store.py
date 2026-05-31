"""In-memory shared blackboard for agent collaboration."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from divine.blackboard.models import (
    Artifact,
    AuditFeedback,
    Event,
    ExecutionResult,
    Fact,
    NodeStatus,
    TaskContext,
    TaskEdge,
    TaskNode,
)
from divine.logger import get_logger


SUCCESS_STATUSES = {"success", "completed"}
_LOG = get_logger("blackboard")


@dataclass
class DynamicTaskGraph:
    nodes: dict[str, TaskNode] = field(default_factory=dict)
    edges: list[TaskEdge] = field(default_factory=list)

    def add_node(self, node: TaskNode) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"Task node already exists: {node.node_id}")
        self.nodes[node.node_id] = node

    def add_edge(self, edge: TaskEdge) -> None:
        if edge.from_node not in self.nodes:
            raise KeyError(f"Missing source node: {edge.from_node}")
        if edge.to_node not in self.nodes:
            raise KeyError(f"Missing target node: {edge.to_node}")
        if edge.edge_type == "dependency" and edge.from_node not in self.nodes[edge.to_node].dependencies:
            self.nodes[edge.to_node].dependencies.append(edge.from_node)
        if not any(
            item.from_node == edge.from_node
            and item.to_node == edge.to_node
            and item.edge_type == edge.edge_type
            for item in self.edges
        ):
            self.edges.append(edge)

    def update_status(self, node_id: str, status: NodeStatus, **metadata: Any) -> None:
        node = self.nodes[node_id]
        node.status = status
        node.metadata.update({key: value for key, value in metadata.items() if value is not None})

    def completed_node_ids(self) -> set[str]:
        return {node_id for node_id, node in self.nodes.items() if node.status in SUCCESS_STATUSES}

    def executable_nodes(self) -> list[TaskNode]:
        completed = self.completed_node_ids()
        return [
            node
            for node in self.nodes.values()
            if node.can_execute_after(completed)
        ]

    def children_of(self, node_id: str, *, edge_types: Iterable[str] | None = None) -> list[TaskNode]:
        allowed = set(edge_types) if edge_types else None
        return [
            self.nodes[edge.to_node]
            for edge in self.edges
            if edge.from_node == node_id and (allowed is None or edge.edge_type in allowed)
        ]

    def descendants_of(self, node_id: str) -> list[TaskNode]:
        descendants: list[TaskNode] = []
        seen: set[str] = set()
        queue = [edge.to_node for edge in self.edges if edge.from_node == node_id]
        while queue:
            current = queue.pop(0)
            if current in seen or current not in self.nodes:
                continue
            seen.add(current)
            descendants.append(self.nodes[current])
            queue.extend(edge.to_node for edge in self.edges if edge.from_node == current)
        return descendants

    def has_task_type(self, task_type: str, *, target: str | None = None) -> bool:
        return any(
            node.task_type == task_type and (target is None or node.target == target)
            for node in self.nodes.values()
            if node.status != "pruned"
        )


@dataclass
class IntelligenceStore:
    target_profile: dict[str, Any] = field(default_factory=dict)
    confirmed_facts: list[Fact] = field(default_factory=list)
    candidate_facts: list[Mapping[str, Any]] = field(default_factory=list)
    rejected_facts: list[Mapping[str, Any]] = field(default_factory=list)
    vulnerabilities: list[Mapping[str, Any]] = field(default_factory=list)
    credentials: list[Mapping[str, Any]] = field(default_factory=list)
    sessions: list[Mapping[str, Any]] = field(default_factory=list)
    services: list[Mapping[str, Any]] = field(default_factory=list)
    technologies: list[Mapping[str, Any]] = field(default_factory=list)

    def add_confirmed_fact(self, fact: Fact) -> bool:
        for existing in self.confirmed_facts:
            if (
                existing.type == fact.type
                and existing.target == fact.target
                and existing.key == fact.key
                and existing.value == fact.value
            ):
                return False
        self.confirmed_facts.append(fact)
        if fact.type == "service":
            self.services.append({"target": fact.target, "key": fact.key, "value": fact.value})
        if fact.type == "technology":
            self.technologies.append({"target": fact.target, "key": fact.key, "value": fact.value})
        return True


@dataclass
class SharedBlackboard:
    context: TaskContext
    graph: DynamicTaskGraph = field(default_factory=DynamicTaskGraph)
    audit_feedback: list[AuditFeedback] = field(default_factory=list)
    execution_results: list[ExecutionResult] = field(default_factory=list)
    intelligence: IntelligenceStore = field(default_factory=IntelligenceStore)
    operation_traces: list[Mapping[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    event_log: list[Event] = field(default_factory=list)
    _counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def next_id(self, prefix: str) -> str:
        while True:
            self._counters[prefix] += 1
            candidate = f"{prefix}_{self._counters[prefix]:03d}"
            if prefix == "task" and candidate in self.graph.nodes:
                continue
            if prefix == "event" and any(event.event_id == candidate for event in self.event_log):
                continue
            if prefix == "fact" and any(fact.fact_id == candidate for fact in self.intelligence.confirmed_facts):
                continue
            return candidate

    def add_node(self, node: TaskNode, *, event_message: str | None = None) -> None:
        self.graph.add_node(node)
        self.record_event(
            "TaskCreated",
            event_message or f"Created task node {node.node_id}",
            {"node_id": node.node_id, "task_type": node.task_type},
        )

    def add_edge(self, edge: TaskEdge) -> None:
        self.graph.add_edge(edge)
        _LOG.debug(
            "Task edge stored task_id={} from_node={} to_node={} edge_type={} reason={}",
            self.context.task_id,
            edge.from_node,
            edge.to_node,
            edge.edge_type,
            edge.reason,
        )

    def add_audit_feedback(self, feedback: AuditFeedback) -> list[str]:
        if any(item.feedback_id == feedback.feedback_id for item in self.audit_feedback):
            return []
        self.audit_feedback.append(feedback)
        new_fact_ids: list[str] = []
        for payload in feedback.audit_result.confirmed_facts:
            fact = Fact.from_mapping(
                self.next_id("fact"),
                payload,
                source=feedback.feedback_id,
                evidence_refs=feedback.audit_result.evidence_refs,
            )
            if self.intelligence.add_confirmed_fact(fact):
                new_fact_ids.append(fact.fact_id)
                self.record_event(
                    "FactConfirmed",
                    f"Confirmed fact {fact.fact_id}",
                    {"fact_id": fact.fact_id, "node_id": feedback.node_id},
                )
        self.record_event(
            "AuditCompleted",
            f"Audit completed for {feedback.node_id}",
            {
                "feedback_id": feedback.feedback_id,
                "node_id": feedback.node_id,
                "status": feedback.task_judgement.status,
            },
        )
        _LOG.info(
            "Audit feedback stored task_id={} feedback_id={} node_id={} status={} failure_level={} new_fact_ids={} evidence_refs={}",
            self.context.task_id,
            feedback.feedback_id,
            feedback.node_id,
            feedback.task_judgement.status,
            feedback.failure_attribution.level,
            new_fact_ids,
            feedback.audit_result.evidence_refs,
        )
        return new_fact_ids

    def add_artifact(self, artifact: Artifact) -> None:
        if artifact.artifact_id in self.artifacts:
            raise ValueError(f"Artifact already exists: {artifact.artifact_id}")
        self.artifacts[artifact.artifact_id] = artifact
        _LOG.debug(
            "Artifact stored task_id={} artifact_id={} artifact_type={} source={} node_id={}",
            self.context.task_id,
            artifact.artifact_id,
            artifact.artifact_type,
            artifact.source,
            artifact.node_id,
        )

    def add_execution_result(self, result: ExecutionResult) -> None:
        self.execution_results.append(result)
        if result.node_id in self.graph.nodes:
            node = self.graph.nodes[result.node_id]
            node.status = result.status if result.status != "needs_more_information" else "blocked"
            node.evidence_refs = _merge_unique(node.evidence_refs, result.evidence_refs)
        self.record_event(
            "ExecutionFinished",
            f"Execution finished for {result.node_id}",
            {
                "execution_id": result.execution_id,
                "node_id": result.node_id,
                "executor": result.executor,
                "status": result.status,
            },
        )
        _LOG.info(
            "Execution result stored task_id={} execution_id={} node_id={} executor={} status={} evidence_refs={} raw_output_refs={} errors={} confidence={}",
            self.context.task_id,
            result.execution_id,
            result.node_id,
            result.executor,
            result.status,
            result.evidence_refs,
            result.raw_output_refs,
            result.errors,
            result.confidence,
        )

    def record_event(
        self,
        event_type: str,
        message: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Event:
        event = Event(
            event_id=self.next_id("event"),
            event_type=event_type,
            message=message,
            payload=dict(payload or {}),
        )
        self.event_log.append(event)
        _LOG.info(
            "Blackboard event task_id={} event_id={} event_type={} message={} payload={}",
            self.context.task_id,
            event.event_id,
            event.event_type,
            event.message,
            dict(event.payload),
        )
        return event


def _merge_unique(left: list[str], right: list[str]) -> list[str]:
    merged = list(left)
    for item in right:
        if item not in merged:
            merged.append(item)
    return merged
