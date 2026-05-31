"""Evaluator Agent backed by LLM semantic audit."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Mapping, Protocol

from divine.blackboard import (
    Artifact,
    AuditFeedback,
    AuditResult,
    ExecutionResult,
    FailureAttribution,
    PlanningFeedback,
    SharedBlackboard,
    TaskJudgement,
    TaskNode,
)
from divine.llm.types import LLMRequest, Message
from divine.logger import get_logger
from divine.prompts import PromptRenderer


class LLMGenerator(Protocol):
    def generate(self, request: LLMRequest) -> Any:
        ...


class EvaluatorAgent:
    """Audits executor output with an LLM and writes feedback to the blackboard."""

    name = "evaluator_agent"

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
        self._log = get_logger("evaluator")

    def audit(
        self,
        *,
        node: TaskNode,
        execution_result: ExecutionResult,
        blackboard: SharedBlackboard,
    ) -> AuditFeedback:
        if node.node_id != execution_result.node_id:
            raise ValueError(
                f"Execution result node_id={execution_result.node_id} does not match node_id={node.node_id}"
            )

        request = self._build_request(node=node, execution_result=execution_result, blackboard=blackboard)
        response = self.llm_client.generate(request)
        payload = _parse_json_object(str(response.content))
        feedback = self._feedback_from_payload(payload, node=node, blackboard=blackboard)
        blackboard.add_audit_feedback(feedback)
        self._log.info(
            "Evaluator audit completed task_id={} node_id={} execution_id={} feedback_id={} status={} failure_level={} recommended_strategy={} evidence_refs={}",
            blackboard.context.task_id,
            node.node_id,
            execution_result.execution_id,
            feedback.feedback_id,
            feedback.task_judgement.status,
            feedback.failure_attribution.level,
            feedback.planning_feedback.recommended_strategy,
            feedback.audit_result.evidence_refs,
        )
        return feedback

    def _build_request(
        self,
        *,
        node: TaskNode,
        execution_result: ExecutionResult,
        blackboard: SharedBlackboard,
    ) -> LLMRequest:
        rendered = self.renderer.render(
            template_id="agents.evaluator.audit",
            variables={
                "target": node.target or blackboard.context.target,
                "node": _json_dumps(_node_payload(node)),
                "success_criteria": node.success_criteria,
                "execution_result": _json_dumps(_execution_payload(execution_result)),
                "evidence": _evidence_payload(execution_result.evidence_refs, blackboard),
                "blackboard_summary": _json_dumps(_blackboard_summary(blackboard)),
            },
            agent=self.name,
        )
        role = self.renderer.render(template_id="agents.evaluator.role", variables={}, agent=self.name)
        contract = self.renderer.render(
            template_id="agents.evaluator.output_contract",
            variables={},
            agent=self.name,
        )
        return LLMRequest(
            messages=[Message("user", rendered.content)],
            system=f"{role.content}\n\n{contract.content}",
            temperature=0,
            max_tokens=self.max_tokens,
            trace_id=rendered.trace_id,
            agent=self.name,
            prompt_trace=rendered.as_trace(),
            trace_metadata={
                "task_id": blackboard.context.task_id,
                "node_id": node.node_id,
                "execution_id": execution_result.execution_id,
                "phase": "evaluator_audit",
                "system_templates": ["agents.evaluator.role", "agents.evaluator.output_contract"],
            },
        )

    def _feedback_from_payload(
        self,
        payload: Mapping[str, Any],
        *,
        node: TaskNode,
        blackboard: SharedBlackboard,
    ) -> AuditFeedback:
        payload_node_id = payload.get("node_id")
        if isinstance(payload_node_id, str) and payload_node_id and payload_node_id != node.node_id:
            raise ValueError(
                f"Evaluator LLM response node_id={payload_node_id} does not match node_id={node.node_id}"
            )
        task_judgement = _mapping(payload.get("task_judgement"))
        audit_result = _mapping(payload.get("audit_result"))
        criteria_audit = _mapping_list(payload.get("criteria_audit"))
        failure_attribution = _mapping(payload.get("failure_attribution"))
        planning_feedback = _mapping(payload.get("planning_feedback"))
        evidence_refs = _valid_evidence_refs(_string_list(audit_result.get("evidence_refs")), blackboard)
        state_updates = _mapping_list(audit_result.get("state_updates"))
        if criteria_audit:
            state_updates.append({"field": "criteria_audit", "value": criteria_audit})

        feedback_id = payload.get("feedback_id")
        if not isinstance(feedback_id, str) or not feedback_id:
            feedback_id = blackboard.next_id("fb")

        return AuditFeedback(
            feedback_id=feedback_id,
            node_id=node.node_id,
            task_judgement=TaskJudgement(
                status=_status(task_judgement.get("status")),
                completion_score=_float(task_judgement.get("completion_score")),
                confidence=_float(task_judgement.get("confidence")),
            ),
            audit_result=AuditResult(
                confirmed_facts=_facts_with_valid_evidence(
                    _mapping_list(audit_result.get("confirmed_facts")),
                    evidence_refs,
                    blackboard,
                ),
                candidate_findings=_mapping_list(audit_result.get("candidate_findings")),
                vulnerabilities=_mapping_list(audit_result.get("vulnerabilities")),
                credentials=_mapping_list(audit_result.get("credentials")),
                sessions=_mapping_list(audit_result.get("sessions")),
                evidence_refs=evidence_refs,
                state_updates=state_updates,
            ),
            failure_attribution=FailureAttribution(
                level=_failure_level(failure_attribution.get("level")),
                primary_cause=_optional_str(failure_attribution.get("primary_cause")),
                secondary_causes=_string_list(failure_attribution.get("secondary_causes")),
                confidence=_float(failure_attribution.get("confidence")),
                reason=_optional_str(failure_attribution.get("reason")),
            ),
            planning_feedback=PlanningFeedback(
                recommended_strategy=_strategy(planning_feedback.get("recommended_strategy")),
                next_focus=_optional_str(planning_feedback.get("next_focus")),
                invalidated_hypothesis=_optional_str(planning_feedback.get("invalidated_hypothesis")),
                should_terminate=bool(planning_feedback.get("should_terminate") or False),
            ),
        )


def _node_payload(node: TaskNode) -> dict[str, Any]:
    return asdict(node)


def _execution_payload(execution_result: ExecutionResult) -> dict[str, Any]:
    return asdict(execution_result)


def _blackboard_summary(blackboard: SharedBlackboard) -> dict[str, Any]:
    return {
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


def _evidence_payload(evidence_refs: list[str], blackboard: SharedBlackboard) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for ref in evidence_refs:
        artifact = blackboard.artifacts.get(ref)
        if not artifact:
            continue
        evidence.append(
            {
                "id": ref,
                "summary": _artifact_summary(artifact),
                "content": _json_dumps(artifact.content),
            }
        )
    return evidence


def _artifact_summary(artifact: Artifact) -> str:
    status = artifact.content.get("status")
    tool_name = artifact.content.get("tool_name") or artifact.source
    error = artifact.content.get("error")
    if error:
        return f"{tool_name} {status}: {error}"
    return f"{tool_name} {status}"


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


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
            raise ValueError("Evaluator LLM response did not contain a JSON object") from None
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, Mapping):
        raise ValueError("Evaluator LLM response must be a JSON object")
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


def _float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(float(value), 1.0))
    return 0.0


def _status(value: Any) -> str:
    status = str(value) if value else "uncertain"
    allowed = {"success", "failed", "partial", "uncertain", "blocked"}
    return status if status in allowed else "uncertain"


def _failure_level(value: Any) -> str:
    level = str(value) if value else "none"
    allowed = {
        "none",
        "execution_failure",
        "cognitive_failure",
        "strategic_failure",
        "constraint_failure",
        "insufficient_evidence",
    }
    return level if level in allowed else "cognitive_failure"


def _strategy(value: Any) -> str | None:
    if value is None:
        return None
    strategy = str(value)
    allowed = {"expand", "regenerate_node", "replan_branch"}
    return strategy if strategy in allowed else None


def _valid_evidence_refs(evidence_refs: list[str], blackboard: SharedBlackboard) -> list[str]:
    return [ref for ref in evidence_refs if ref in blackboard.artifacts]


def _facts_with_valid_evidence(
    facts: list[Mapping[str, Any]],
    default_evidence_refs: list[str],
    blackboard: SharedBlackboard,
) -> list[Mapping[str, Any]]:
    normalized: list[Mapping[str, Any]] = []
    for fact in facts:
        refs = _valid_evidence_refs(_string_list(fact.get("evidence_refs")), blackboard)
        if not refs:
            refs = default_evidence_refs
        if not refs:
            continue
        normalized.append({**dict(fact), "evidence_refs": refs})
    return normalized
