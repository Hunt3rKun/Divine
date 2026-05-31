import json
from dataclasses import asdict
from typing import Any, Mapping

from loguru import logger

from divine.blackboard import Blackboard
from divine.llm.base import LLMMessage
from divine.llm.router import LLMRouter
from divine.models.audit import (
    AuditFeedback,
    AuditResult,
    FailureAttribution,
    PlanningFeedback,
    TaskJudgement,
)
from divine.models.task import TaskNode


class EvaluatorAgent:
    """LLM-backed semantic audit for executor results."""

    def __init__(
        self,
        router: LLMRouter,
        *,
        model: str,
    ) -> None:
        self._router = router
        self._model = model

    async def audit(
        self,
        *,
        task: TaskNode,
        execution_result: Mapping[str, Any],
        blackboard: Blackboard,
    ) -> AuditFeedback:
        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=_build_user_prompt(task, execution_result, blackboard),
            ),
        ]
        response = await self._router.chat(
            self._model,
            messages,
            agent="evaluator_agent",
            trace_metadata={
                "task_id": task.id,
                "phase": "evaluator_audit",
            },
        )
        payload = _parse_json_object(response.content)
        feedback = feedback_from_payload(
            payload,
            task_id=task.id,
            available_evidence_refs=_available_evidence_refs(execution_result),
        )
        persist_feedback(blackboard, feedback)
        logger.info(
            "Evaluator audit completed task_id={} feedback_id={} status={} failure_level={} strategy={}",
            task.id,
            feedback.feedback_id,
            feedback.task_judgement.status,
            feedback.failure_attribution.level,
            feedback.planning_feedback.recommended_strategy,
        )
        return feedback


def feedback_from_payload(
    payload: Mapping[str, Any],
    *,
    task_id: str,
    available_evidence_refs: set[str] | None = None,
) -> AuditFeedback:
    payload_task_id = payload.get("task_id") or payload.get("node_id")
    if isinstance(payload_task_id, str) and payload_task_id and payload_task_id != task_id:
        raise ValueError(f"Evaluator response task_id={payload_task_id} does not match task_id={task_id}")

    task_judgement = _mapping(payload.get("task_judgement"))
    audit_result = _mapping(payload.get("audit_result"))
    failure_attribution = _mapping(payload.get("failure_attribution"))
    planning_feedback = _mapping(payload.get("planning_feedback"))
    criteria_audit = _mapping_list(payload.get("criteria_audit"))
    evidence_refs = _valid_evidence_refs(
        _string_list(audit_result.get("evidence_refs")),
        available_evidence_refs,
    )
    state_updates = _mapping_list(audit_result.get("state_updates"))
    if criteria_audit:
        state_updates.append({"field": "criteria_audit", "value": criteria_audit})

    feedback_id = payload.get("feedback_id")
    if not isinstance(feedback_id, str) or not feedback_id:
        feedback_id = f"audit_{task_id}"

    return AuditFeedback(
        feedback_id=feedback_id,
        task_id=task_id,
        task_judgement=TaskJudgement(
            status=_status(task_judgement.get("status")),
            completion_score=_float(task_judgement.get("completion_score")),
            confidence=_float(task_judgement.get("confidence")),
        ),
        audit_result=AuditResult(
            confirmed_facts=_facts_with_valid_evidence(
                _mapping_list(audit_result.get("confirmed_facts")),
                evidence_refs,
                available_evidence_refs,
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


def persist_feedback(blackboard: Blackboard, feedback: AuditFeedback) -> None:
    blackboard.write(
        "reflections",
        feedback.feedback_id,
        asdict(feedback),
        source="evaluator",
    )
    for index, fact in enumerate(feedback.audit_result.confirmed_facts, start=1):
        key = str(fact.get("key") or fact.get("type") or f"fact_{index}")
        blackboard.write("findings", f"{feedback.task_id}:{key}", dict(fact), source=feedback.feedback_id)
    for index, credential in enumerate(feedback.audit_result.credentials, start=1):
        blackboard.write(
            "credentials",
            f"{feedback.task_id}:credential_{index}",
            dict(credential),
            source=feedback.feedback_id,
        )


def _build_user_prompt(task: TaskNode, execution_result: Mapping[str, Any], blackboard: Blackboard) -> str:
    payload = {
        "task": {
            "id": task.id,
            "description": task.description,
            "phase": task.phase.value,
            "executor_type": task.executor_type.value,
            "dependencies": task.dependencies,
            "status": task.status.value,
        },
        "execution_result": dict(execution_result),
        "blackboard_summary": blackboard.summary(
            sections=["hosts", "ports", "credentials", "findings", "reflections"],
        ),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _available_evidence_refs(execution_result: Mapping[str, Any]) -> set[str]:
    refs: set[str] = set()
    for key in ("evidence_refs", "raw_output_refs"):
        refs.update(_string_list(execution_result.get(key)))
    bb_writes = execution_result.get("bb_writes")
    if isinstance(bb_writes, Mapping):
        for section, keys in bb_writes.items():
            for item in _string_list(keys):
                refs.add(f"{section}:{item}")
    return refs


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


def _valid_evidence_refs(evidence_refs: list[str], available_evidence_refs: set[str] | None) -> list[str]:
    if available_evidence_refs is None:
        return evidence_refs
    return [ref for ref in evidence_refs if ref in available_evidence_refs]


def _facts_with_valid_evidence(
    facts: list[Mapping[str, Any]],
    default_evidence_refs: list[str],
    available_evidence_refs: set[str] | None,
) -> list[Mapping[str, Any]]:
    normalized: list[Mapping[str, Any]] = []
    for fact in facts:
        refs = _string_list(fact.get("evidence_refs")) or default_evidence_refs
        refs = _valid_evidence_refs(refs, available_evidence_refs)
        if not refs:
            continue
        item = dict(fact)
        item["evidence_refs"] = refs
        normalized.append(item)
    return normalized


_SYSTEM_PROMPT = """你是 Divine 的评估智能体，负责审计执行智能体的输出。

请只返回 JSON 对象，字段包括：
- task_id: 被审计任务 ID
- task_judgement: {status, completion_score, confidence}
- audit_result: {confirmed_facts, candidate_findings, vulnerabilities, credentials, sessions, evidence_refs, state_updates}
- failure_attribution: {level, primary_cause, secondary_causes, confidence, reason}
- planning_feedback: {recommended_strategy, next_focus, invalidated_hypothesis, should_terminate}

必须依据 evidence_refs 判断事实是否可确认；缺少证据时不要把候选事实升级为 confirmed_facts。
"""

