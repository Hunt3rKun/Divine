"""Planner Agent for dynamic task DAG creation and evolution."""

from __future__ import annotations

import json
from typing import Any, Mapping, Protocol

from divine.blackboard.models import (
    AuditFeedback,
    PlannerOperation,
    PlannerResult,
    PlanningStrategy,
    TaskContext,
    TaskEdge,
    TaskNode,
)
from divine.blackboard.store import SharedBlackboard
from divine.llm.types import LLMRequest, Message
from divine.logger import get_logger
from divine.prompts import PromptRenderer


class LLMGenerator(Protocol):
    def generate(self, request: LLMRequest) -> Any:
        ...


ALLOWED_PLANNER_OPERATIONS = {"create_node", "update_node_status"}
ALLOWED_PLANNER_STATUS_UPDATES = {"completed", "blocked", "pruned", "pending"}
ALLOWED_PLANNER_TASK_TYPES = {
    "asset_discovery",
    "port_check",
    "service_detection",
    "web_fingerprint",
    "web_path_discovery",
    "web_rule_check",
    "response_header_analysis",
    "page_title_extraction",
    "static_resource_analysis",
    "service_detection_validation",
    "web_fingerprint_validation",
    "host_info",
}
ALLOWED_PLANNER_EXECUTORS = {"recon_agent", "web_agent", "host_agent"}
VALID_DEPENDENCY_STATUSES = {"pending", "success", "completed"}
MAX_CREATED_NODES_PER_DECISION = 3


class PlannerAgent:
    """Maintains the latest shared DAG from structured blackboard feedback.

    The engineering-level evolution strategies are intentionally limited to:
    expand after success, replan a failed branch, or regenerate a failed node.
    Pruning is modeled as a side effect of the two failure strategies.
    """

    name = "planner_agent"

    def __init__(
        self,
        llm_client: LLMGenerator,
        *,
        renderer: PromptRenderer | None = None,
        max_tokens: int = 4096,
    ) -> None:
        self.llm_client = llm_client
        self.renderer = renderer or PromptRenderer()
        self.max_tokens = max_tokens
        self._log = get_logger("planner")

    def generate_initial_dag(self, context: TaskContext, blackboard: SharedBlackboard) -> PlannerResult:
        return self._generate_initial_dag_with_llm(context, blackboard)

    def evolve_dag(self, feedback: AuditFeedback, blackboard: SharedBlackboard) -> PlannerResult:
        return self._evolve_dag_with_llm(feedback, blackboard)

    def _select_strategy(self, feedback: AuditFeedback) -> PlanningStrategy:
        level = feedback.failure_attribution.level
        status = feedback.task_judgement.status
        if level == "execution_failure":
            return "regenerate_node"
        if level in {"cognitive_failure", "strategic_failure", "constraint_failure", "insufficient_evidence"}:
            return "replan_branch"
        if status == "success":
            return "expand"
        return "replan_branch"

    def _expand_from_success(self, feedback: AuditFeedback, blackboard: SharedBlackboard) -> PlannerResult:
        trigger = blackboard.graph.nodes[feedback.node_id]
        trigger.status = "success"
        trigger.evidence_refs = _merge_unique(trigger.evidence_refs, feedback.audit_result.evidence_refs)

        added: list[str] = []
        operations: list[PlannerOperation] = [
            PlannerOperation("update_node_status", trigger.node_id, "Evaluator judged the node successful.")
        ]
        for fact in feedback.audit_result.confirmed_facts:
            for node in self._nodes_for_fact(fact, trigger, blackboard):
                blackboard.add_node(node)
                blackboard.add_edge(
                    TaskEdge(
                        from_node=trigger.node_id,
                        to_node=node.node_id,
                        edge_type="dependency",
                        reason=f"Expanded from confirmed fact type={fact.get('type')}",
                    )
                )
                added.append(node.node_id)
                operations.append(
                    PlannerOperation("create_node", node.node_id, f"Confirmed fact supports {node.task_type}.")
                )

        reason = "Successful task produced confirmed facts with planning value."
        if not added:
            trigger.status = "completed"
            reason = "Successful task produced no new graph expansion."
            operations.append(
                PlannerOperation("update_node_status", trigger.node_id, "No new unique task emerged; archive node as completed.")
            )

        return self._finalize(
            blackboard,
            strategy="expand",
            trigger_node=trigger.node_id,
            reason=reason,
            operations=operations,
            added_nodes=added,
            updated_nodes=[trigger.node_id],
        )

    def _nodes_for_fact(
        self,
        fact: Mapping[str, Any],
        trigger: TaskNode,
        blackboard: SharedBlackboard,
    ) -> list[TaskNode]:
        fact_type = str(fact.get("type") or "")
        value = str(fact.get("value") or "").lower()
        target = str(fact.get("target") or trigger.target or blackboard.context.target)

        candidates: list[TaskNode] = []
        if fact_type == "service" and value in {"http", "https", "web"}:
            if not blackboard.graph.has_task_type("web_fingerprint", target=target):
                candidates.append(
                    TaskNode(
                        node_id=blackboard.next_id("task"),
                        task_type="web_fingerprint",
                        description=f"Identify Web technology and response characteristics for {target}.",
                        dependencies=[trigger.node_id],
                        risk_level="low",
                        success_criteria=[
                            "Collect response status, headers, title, and basic static indicators.",
                            "Produce evidence references for every fingerprint claim.",
                            "Output candidate technology facts without treating them as confirmed.",
                        ],
                        assigned_executor="web_agent",
                        created_from=trigger.node_id,
                        target=target,
                    )
                )
            if not blackboard.graph.has_task_type("web_path_discovery", target=target):
                candidates.append(
                    TaskNode(
                        node_id=blackboard.next_id("task"),
                        task_type="web_path_discovery",
                        description=f"Probe common Web paths for {target}.",
                        dependencies=[trigger.node_id],
                        risk_level="low",
                        success_criteria=[
                            "Record status codes and evidence for discovered paths.",
                            "Summarize discovered paths and unresolved observations.",
                        ],
                        assigned_executor="web_agent",
                        created_from=trigger.node_id,
                        target=target,
                    )
                )
        elif fact_type == "technology":
            if not blackboard.graph.has_task_type("web_rule_check", target=target):
                candidates.append(
                    TaskNode(
                        node_id=blackboard.next_id("task"),
                        task_type="web_rule_check",
                        description=f"Run rule checks for confirmed technology on {target}.",
                        dependencies=[trigger.node_id],
                        risk_level="low",
                        success_criteria=[
                            "Tie every candidate finding to evidence.",
                            "Separate suspected findings from verified vulnerabilities.",
                        ],
                        assigned_executor="web_agent",
                        created_from=trigger.node_id,
                        target=target,
                    )
                )
        return candidates

    def _regenerate_node(self, feedback: AuditFeedback, blackboard: SharedBlackboard) -> PlannerResult:
        original = blackboard.graph.nodes[feedback.node_id]
        original.status = feedback.task_judgement.status
        original.attempt_count += 1
        original.evidence_refs = _merge_unique(original.evidence_refs, feedback.audit_result.evidence_refs)
        pruned = self._prune_invalid_descendants(feedback, blackboard)

        operations = [
            PlannerOperation("update_node_status", original.node_id, "Evaluator judged this attempt unsuccessful."),
            *[
                PlannerOperation("update_node_status", node_id, "Prune stale downstream node while regenerating failed node.")
                for node_id in pruned
            ],
        ]
        added: list[str] = []
        reason = "Execution-layer failure with retry budget available."

        if original.attempt_count >= original.max_attempts:
            original.status = "blocked"
            original.metadata["blocked_reason"] = "Maximum attempts reached before regeneration."
            operations.append(PlannerOperation("update_node_status", original.node_id, "Block node after exhausted attempts."))
            reason = "Execution failed and retry budget is exhausted."
        else:
            replacement = TaskNode(
                node_id=blackboard.next_id("task"),
                task_type=original.task_type,
                description=f"Alternative attempt for: {original.description}",
                dependencies=list(original.dependencies),
                edge_type="alternative",
                risk_level=original.risk_level,
                success_criteria=list(original.success_criteria),
                assigned_executor=original.assigned_executor,
                max_attempts=original.max_attempts,
                created_from=original.node_id,
                target=original.target,
                metadata={
                    "regenerated_from": original.node_id,
                    "regeneration_reason": feedback.failure_attribution.reason,
                },
            )
            blackboard.add_node(replacement)
            blackboard.add_edge(
                TaskEdge(
                    from_node=original.node_id,
                    to_node=replacement.node_id,
                    edge_type="alternative",
                    reason=feedback.failure_attribution.reason
                    or "Execution-layer failure requires an alternative attempt.",
                )
            )
            added.append(replacement.node_id)
            operations.append(
                PlannerOperation("create_node", replacement.node_id, "Create alternative node for execution failure.")
            )

        return self._finalize(
            blackboard,
            strategy="regenerate_node",
            trigger_node=original.node_id,
            reason=reason,
            operations=operations,
            added_nodes=added,
            updated_nodes=[original.node_id],
            pruned_nodes=pruned,
        )

    def _replan_branch(self, feedback: AuditFeedback, blackboard: SharedBlackboard) -> PlannerResult:
        trigger = blackboard.graph.nodes[feedback.node_id]
        trigger.status = feedback.task_judgement.status
        trigger.evidence_refs = _merge_unique(trigger.evidence_refs, feedback.audit_result.evidence_refs)
        pruned = self._prune_invalid_descendants(feedback, blackboard)

        if feedback.failure_attribution.level == "constraint_failure" or feedback.planning_feedback.should_terminate:
            trigger.status = "blocked"
            trigger.metadata["blocked_reason"] = feedback.failure_attribution.reason or "Constraint failure."
            return self._finalize(
                blackboard,
                strategy="replan_branch",
                trigger_node=trigger.node_id,
                reason="Failure blocks the current branch; invalid downstream nodes were pruned.",
                operations=[
                    PlannerOperation("update_node_status", trigger.node_id, "Block failed branch root."),
                    *[
                        PlannerOperation("update_node_status", node_id, "Prune invalid downstream node during branch replanning.")
                        for node_id in pruned
                    ],
                ],
                updated_nodes=[trigger.node_id],
                pruned_nodes=pruned,
            )

        if feedback.failure_attribution.level == "strategic_failure":
            return self._finalize(
                blackboard,
                strategy="replan_branch",
                trigger_node=trigger.node_id,
                reason="Strategic failure invalidated the current branch; pruning is part of branch replanning.",
                operations=[
                    PlannerOperation("update_node_status", trigger.node_id, "Record strategic failure on branch root."),
                    *[
                        PlannerOperation("update_node_status", node_id, "Prune downstream node invalidated by branch failure.")
                        for node_id in pruned
                    ],
                ],
                updated_nodes=[trigger.node_id],
                pruned_nodes=pruned,
            )

        if feedback.failure_attribution.level == "insufficient_evidence":
            return self._add_validation_as_replan(feedback, blackboard, pruned)

        target = trigger.target or blackboard.context.target
        specs = [
            (
                "response_header_analysis",
                f"Analyze response headers for {target} before repeating higher-level fingerprinting.",
            ),
            (
                "page_title_extraction",
                f"Extract page title and visible page markers for {target}.",
            ),
            (
                "static_resource_analysis",
                f"Inspect static resource references for technology clues on {target}.",
            ),
        ]
        added: list[str] = []
        for task_type, description in specs:
            if blackboard.graph.has_task_type(task_type, target=target):
                continue
            node = TaskNode(
                node_id=blackboard.next_id("task"),
                task_type=task_type,
                description=description,
                dependencies=list(trigger.dependencies),
                edge_type="alternative",
                risk_level="low",
                success_criteria=[
                    "Collect a narrow, auditable observation for the blocked branch.",
                    "Produce evidence references for any candidate fact.",
                ],
                assigned_executor="web_agent",
                created_from=trigger.node_id,
                target=target,
                metadata={"replanned_from": trigger.node_id},
            )
            blackboard.add_node(node)
            blackboard.add_edge(
                TaskEdge(
                    from_node=trigger.node_id,
                    to_node=node.node_id,
                    edge_type="alternative",
                    reason="Cognitive failure requires lower-level observations before retrying branch strategy.",
                )
            )
            added.append(node.node_id)

        return self._finalize(
            blackboard,
            strategy="replan_branch",
            trigger_node=trigger.node_id,
            reason="Cognitive failure suggests the local branch needs decomposed observations.",
            operations=[
                PlannerOperation("update_node_status", trigger.node_id, "Record cognitive failure on original branch."),
                *[
                    PlannerOperation("create_node", node_id, "Add decomposed observation task.")
                    for node_id in added
                ],
                *[
                    PlannerOperation("update_node_status", node_id, "Prune invalid downstream node during branch replanning.")
                    for node_id in pruned
                ],
            ],
            added_nodes=added,
            updated_nodes=[trigger.node_id],
            pruned_nodes=pruned,
        )

    def _add_validation_as_replan(
        self,
        feedback: AuditFeedback,
        blackboard: SharedBlackboard,
        pruned_nodes: list[str],
    ) -> PlannerResult:
        trigger = blackboard.graph.nodes[feedback.node_id]
        node = TaskNode(
            node_id=blackboard.next_id("task"),
            task_type=f"{trigger.task_type}_validation",
            description=f"Collect additional evidence to validate result for {trigger.description}",
            dependencies=[trigger.node_id] if trigger.status in {"success", "completed"} else list(trigger.dependencies),
            edge_type="validation",
            risk_level=trigger.risk_level,
            success_criteria=[
                "Gather enough evidence to resolve the prior inconclusive audit.",
                "Reference all raw or structured outputs as evidence.",
            ],
            assigned_executor=trigger.assigned_executor,
            created_from=trigger.node_id,
            target=trigger.target,
            metadata={"validation_for": trigger.node_id},
        )
        blackboard.add_node(node)
        blackboard.add_edge(
            TaskEdge(
                from_node=trigger.node_id,
                to_node=node.node_id,
                edge_type="validation",
                reason="Evaluator reported insufficient evidence.",
            )
        )
        return self._finalize(
            blackboard,
            strategy="replan_branch",
            trigger_node=trigger.node_id,
            reason="Audit could not make a confident judgement; add validation as branch replanning.",
            operations=[
                PlannerOperation("update_node_status", trigger.node_id, "Record inconclusive or partial audit."),
                PlannerOperation("create_node", node.node_id, "Add validation task for insufficient evidence."),
                *[
                    PlannerOperation("update_node_status", node_id, "Prune invalid downstream node during branch replanning.")
                    for node_id in pruned_nodes
                ],
            ],
            added_nodes=[node.node_id],
            updated_nodes=[trigger.node_id],
            pruned_nodes=pruned_nodes,
        )

    def _prune_invalid_descendants(
        self,
        feedback: AuditFeedback,
        blackboard: SharedBlackboard,
    ) -> list[str]:
        trigger = blackboard.graph.nodes[feedback.node_id]
        invalidated = feedback.planning_feedback.invalidated_hypothesis or feedback.failure_attribution.reason
        should_prune_all = feedback.failure_attribution.level == "strategic_failure"
        pruned: list[str] = []
        for node in blackboard.graph.descendants_of(trigger.node_id):
            if node.status in {"success", "completed", "pruned"}:
                continue
            is_hypothesis_dependent = node.edge_type == "hypothesis" or bool(node.metadata.get("depends_on_failed_branch"))
            if not should_prune_all and not is_hypothesis_dependent:
                continue
            node.status = "pruned"
            node.metadata.update(
                {
                    "pruned_by": trigger.node_id,
                    "prune_reason": feedback.failure_attribution.reason,
                    "invalidated_hypothesis": invalidated,
                }
            )
            pruned.append(node.node_id)
        return pruned

    def _finalize(
        self,
        blackboard: SharedBlackboard,
        *,
        strategy: str,
        trigger_node: str | None,
        reason: str,
        operations: list[PlannerOperation],
        added_nodes: list[str] | None = None,
        updated_nodes: list[str] | None = None,
        pruned_nodes: list[str] | None = None,
    ) -> PlannerResult:
        blackboard.record_event(
            "DAGEvolved",
            f"DAG evolved with strategy {strategy}",
            {
                "strategy": strategy,
                "trigger_node": trigger_node,
                "added_nodes": list(added_nodes or []),
                "updated_nodes": list(updated_nodes or []),
                "pruned_nodes": list(pruned_nodes or []),
            },
        )
        return PlannerResult(
            status="evolved",
            strategy=strategy,
            rationale=reason,
            operations=operations,
            added_nodes=list(added_nodes or []),
            updated_nodes=list(updated_nodes or []),
            pruned_nodes=list(pruned_nodes or []),
        )

    def _generate_initial_dag_with_llm(
        self,
        context: TaskContext,
        blackboard: SharedBlackboard,
    ) -> PlannerResult:
        if blackboard.graph.nodes:
            return PlannerResult(
                status="planned",
                strategy="noop",
                rationale="Initial DAG already exists.",
            )
        request = self._build_initial_request(context, blackboard)
        response = self.llm_client.generate(request)  # type: ignore[union-attr]
        payload = _parse_json_object(str(response.content))
        self._log.debug(
            "Planner initial decision received task_id={} trace_id={}",
            context.task_id,
            request.trace_id,
        )
        return self._apply_llm_decision(payload, blackboard, default_strategy="initial_plan")

    def _evolve_dag_with_llm(self, feedback: AuditFeedback, blackboard: SharedBlackboard) -> PlannerResult:
        if feedback.node_id not in blackboard.graph.nodes:
            raise KeyError(f"Cannot evolve DAG from unknown node: {feedback.node_id}")
        if not any(item.feedback_id == feedback.feedback_id for item in blackboard.audit_feedback):
            raise ValueError(
                f"Audit feedback {feedback.feedback_id} must be written to the blackboard before planning."
            )
        request = self._build_evolve_request(feedback, blackboard)
        response = self.llm_client.generate(request)  # type: ignore[union-attr]
        payload = _parse_json_object(str(response.content))
        self._log.debug(
            "Planner evolve decision received task_id={} node_id={} trace_id={}",
            blackboard.context.task_id,
            feedback.node_id,
            request.trace_id,
        )
        return self._apply_llm_decision(payload, blackboard, default_strategy=self._select_strategy(feedback))

    def _build_initial_request(self, context: TaskContext, blackboard: SharedBlackboard) -> LLMRequest:
        rendered = self.renderer.render(
            template_id="agents.planner.initial_dag",
            variables={
                "target": context.target,
                "objective": context.goal,
                "blackboard_summary": _blackboard_summary(blackboard),
            },
            agent=self.name,
        )
        return self._request_from_rendered(
            rendered.content,
            rendered.trace_id,
            rendered.as_trace(),
            trace_metadata={"task_id": context.task_id, "phase": "planner_initial"},
        )

    def _build_evolve_request(self, feedback: AuditFeedback, blackboard: SharedBlackboard) -> LLMRequest:
        rendered = self.renderer.render(
            template_id="agents.planner.evolve_dag",
            variables={
                "dag_state": _dag_state(blackboard),
                "execution_summary": _latest_execution_summary(feedback.node_id, blackboard),
                "audit_feedback": _json_dumps(_feedback_payload(feedback)),
            },
            agent=self.name,
        )
        return self._request_from_rendered(
            rendered.content,
            rendered.trace_id,
            rendered.as_trace(),
            trace_metadata={
                "task_id": blackboard.context.task_id,
                "node_id": feedback.node_id,
                "phase": "planner_evolve",
                "feedback_id": feedback.feedback_id,
            },
        )

    def _request_from_rendered(
        self,
        content: str,
        trace_id: str,
        prompt_trace: Mapping[str, Any],
        trace_metadata: Mapping[str, Any] | None = None,
    ) -> LLMRequest:
        role = self.renderer.render(template_id="agents.planner.role", variables={}, agent=self.name)
        contract = self.renderer.render(template_id="agents.planner.output_contract", variables={}, agent=self.name)
        return LLMRequest(
            messages=[Message("user", content)],
            system=f"{role.content}\n\n{contract.content}",
            temperature=0,
            max_tokens=self.max_tokens,
            trace_id=trace_id,
            agent=self.name,
            prompt_trace=prompt_trace,
            trace_metadata={
                "phase": "planner_initial",
                "system_templates": ["agents.planner.role", "agents.planner.output_contract"],
                **dict(trace_metadata or {}),
            },
        )

    def _apply_llm_decision(
        self,
        payload: Mapping[str, Any],
        blackboard: SharedBlackboard,
        *,
        default_strategy: str,
    ) -> PlannerResult:
        operations_payload = _mapping_list(payload.get("dag_operations"))
        create_count = sum(1 for item in operations_payload if item.get("operation") == "create_node")
        if create_count > MAX_CREATED_NODES_PER_DECISION:
            raise ValueError("Planner decision exceeded maximum create_node operations.")

        local_refs: dict[str, str] = {}
        operations: list[PlannerOperation] = []
        added_nodes: list[str] = []
        updated_nodes: list[str] = []
        pruned_nodes: list[str] = []

        for item in operations_payload:
            operation = str(item.get("operation") or "")
            if operation not in ALLOWED_PLANNER_OPERATIONS:
                raise ValueError(f"Unsupported planner operation: {operation}")
            if operation == "create_node":
                local_ref = str(item.get("local_ref") or "")
                if local_ref and local_ref in local_refs:
                    raise ValueError(f"Duplicate planner local_ref: {local_ref}")
                node = self._node_from_operation(item, blackboard, local_refs)
                blackboard.add_node(node)
                self._log.info(
                    "Planner create_node task_id={} node_id={} task_type={} local_ref={} dependencies={} risk_level={}",
                    blackboard.context.task_id,
                    node.node_id,
                    node.task_type,
                    item.get("local_ref"),
                    node.dependencies,
                    node.risk_level,
                )
                for dependency in node.dependencies:
                    if dependency in blackboard.graph.nodes:
                        blackboard.add_edge(
                            TaskEdge(
                                from_node=dependency,
                                to_node=node.node_id,
                                edge_type=node.edge_type,
                                reason=str(item.get("reason") or ""),
                            )
                        )
                if local_ref:
                    local_refs[local_ref] = node.node_id
                added_nodes.append(node.node_id)
                operations.append(
                    PlannerOperation("create_node", node.node_id, str(item.get("reason") or "Create node."))
                )
                continue

            node_id = str(item.get("node_id") or "")
            status = str(item.get("status") or "")
            if node_id not in blackboard.graph.nodes:
                raise KeyError(f"Cannot update unknown node: {node_id}")
            if status not in ALLOWED_PLANNER_STATUS_UPDATES:
                raise ValueError(f"Unsupported planner status update: {status}")
            blackboard.graph.update_status(node_id, status)  # type: ignore[arg-type]
            self._log.info(
                "Planner update_node_status task_id={} node_id={} status={} reason={}",
                blackboard.context.task_id,
                node_id,
                status,
                item.get("reason"),
            )
            updated_nodes.append(node_id)
            if status == "pruned":
                pruned_nodes.append(node_id)
            operations.append(
                PlannerOperation("update_node_status", node_id, str(item.get("reason") or f"Set status to {status}."))
            )

        termination = _mapping(payload.get("termination_decision"))
        should_terminate = bool(termination.get("should_terminate") or payload.get("status") == "terminate")
        if added_nodes and not should_terminate:
            self._validate_added_node_dependencies(added_nodes, blackboard)
        strategy = str(payload.get("strategy") or default_strategy)
        rationale = str(payload.get("rationale") or "Planner LLM decision applied.")
        blackboard.record_event(
            "DAGEvolved",
            f"DAG evolved with strategy {strategy}",
            {
                "strategy": strategy,
                "added_nodes": added_nodes,
                "updated_nodes": updated_nodes,
                "pruned_nodes": pruned_nodes,
            },
        )
        self._log.info(
            "Planner decision applied task_id={} strategy={} status={} added_nodes={} updated_nodes={} pruned_nodes={} should_terminate={}",
            blackboard.context.task_id,
            strategy,
            payload.get("status") or "evolved",
            added_nodes,
            updated_nodes,
            pruned_nodes,
            should_terminate,
        )
        return PlannerResult(
            status=str(payload.get("status") or "evolved"),
            strategy=strategy,
            rationale=rationale,
            operations=operations,
            added_nodes=added_nodes,
            updated_nodes=updated_nodes,
            pruned_nodes=pruned_nodes,
            should_terminate=should_terminate,
        )

    def _node_from_operation(
        self,
        operation: Mapping[str, Any],
        blackboard: SharedBlackboard,
        local_refs: Mapping[str, str],
    ) -> TaskNode:
        node_payload = _mapping(operation.get("node"))
        task_type = str(node_payload.get("task_type") or "")
        if task_type not in ALLOWED_PLANNER_TASK_TYPES:
            raise ValueError(f"Unsupported planner task_type: {task_type}")
        assigned_executor = _optional_str(node_payload.get("assigned_executor"))
        if assigned_executor not in ALLOWED_PLANNER_EXECUTORS:
            raise ValueError(f"Unsupported planner assigned_executor: {assigned_executor}")
        dependencies = [
            local_refs.get(str(item), str(item))
            for item in _string_list(node_payload.get("dependencies"))
        ]
        missing_dependencies = [
            dependency
            for dependency in dependencies
            if dependency not in blackboard.graph.nodes
        ]
        if missing_dependencies:
            raise ValueError(f"Planner node dependencies are missing: {missing_dependencies}")
        invalid_dependencies = [
            {
                "dependency": dependency,
                "dependency_status": blackboard.graph.nodes[dependency].status,
            }
            for dependency in dependencies
            if blackboard.graph.nodes[dependency].status not in VALID_DEPENDENCY_STATUSES
        ]
        if invalid_dependencies:
            raise ValueError(f"Planner node dependencies are non-executable: {invalid_dependencies}")
        return TaskNode(
            node_id=blackboard.next_id("task"),
            task_type=task_type,
            description=str(node_payload.get("description") or ""),
            dependencies=dependencies,
            edge_type=_edge_type(node_payload.get("edge_type")),
            risk_level=_risk_level(node_payload.get("risk_level")),
            success_criteria=_string_list(node_payload.get("success_criteria")),
            assigned_executor=assigned_executor,
            target=_optional_str(node_payload.get("target")) or blackboard.context.target,
            metadata={"planner_local_ref": _optional_str(operation.get("local_ref"))},
        )

    def _validate_added_node_dependencies(
        self,
        added_nodes: list[str],
        blackboard: SharedBlackboard,
    ) -> None:
        invalid: list[dict[str, str]] = []
        for node_id in added_nodes:
            node = blackboard.graph.nodes[node_id]
            for dependency in node.dependencies:
                dependency_node = blackboard.graph.nodes[dependency]
                if dependency_node.status not in VALID_DEPENDENCY_STATUSES:
                    invalid.append(
                        {
                            "node_id": node_id,
                            "dependency": dependency,
                            "dependency_status": dependency_node.status,
                        }
                    )
        if invalid:
            raise ValueError(f"Planner created nodes with non-executable dependencies: {invalid}")


def _merge_unique(left: list[str], right: list[str]) -> list[str]:
    merged = list(left)
    for item in right:
        if item not in merged:
            merged.append(item)
    return merged


def _parse_json_object(content: str) -> Mapping[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Planner LLM response did not contain a JSON object") from None
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, Mapping):
        raise ValueError("Planner LLM response must be a JSON object")
    return payload


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _optional_str(value: Any) -> str | None:
    return str(value) if value not in {None, ""} else None


def _edge_type(value: Any) -> str:
    edge_type = str(value) if value else "dependency"
    return edge_type if edge_type in {"dependency", "hypothesis", "validation", "alternative"} else "dependency"


def _risk_level(value: Any) -> str:
    risk_level = str(value) if value else "low"
    return risk_level if risk_level in {"low", "medium", "high"} else "low"


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _dag_state(blackboard: SharedBlackboard) -> str:
    payload = {
        node_id: {
            "task_type": node.task_type,
            "status": node.status,
            "dependencies": node.dependencies,
            "target": node.target,
        }
        for node_id, node in blackboard.graph.nodes.items()
    }
    return _json_dumps(payload)


def _blackboard_summary(blackboard: SharedBlackboard) -> str:
    payload = {
        "confirmed_facts": [
            {
                "type": fact.type,
                "target": fact.target,
                "key": fact.key,
                "value": fact.value,
            }
            for fact in blackboard.intelligence.confirmed_facts
        ],
        "recent_events": [
            {"event_type": event.event_type, "message": event.message}
            for event in blackboard.event_log[-5:]
        ],
    }
    return _json_dumps(payload)


def _latest_execution_summary(node_id: str, blackboard: SharedBlackboard) -> str:
    for result in reversed(blackboard.execution_results):
        if result.node_id == node_id:
            return _json_dumps(
                {
                    "status": result.status,
                    "summary": result.summary,
                    "errors": result.errors,
                    "evidence_refs": result.evidence_refs,
                }
            )
    return "null"


def _feedback_payload(feedback: AuditFeedback) -> Mapping[str, Any]:
    return {
        "feedback_id": feedback.feedback_id,
        "node_id": feedback.node_id,
        "task_judgement": {
            "status": feedback.task_judgement.status,
            "completion_score": feedback.task_judgement.completion_score,
            "confidence": feedback.task_judgement.confidence,
        },
        "failure_attribution": {
            "level": feedback.failure_attribution.level,
            "primary_cause": feedback.failure_attribution.primary_cause,
            "secondary_causes": feedback.failure_attribution.secondary_causes,
            "confidence": feedback.failure_attribution.confidence,
            "reason": feedback.failure_attribution.reason,
        },
        "planning_feedback": {
            "recommended_strategy": feedback.planning_feedback.recommended_strategy,
            "next_focus": feedback.planning_feedback.next_focus,
            "invalidated_hypothesis": feedback.planning_feedback.invalidated_hypothesis,
            "should_terminate": feedback.planning_feedback.should_terminate,
        },
    }
