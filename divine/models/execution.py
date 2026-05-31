from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class ExecutionResult:
    execution_id: str
    task_id: str
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


def execution_result_from_final_action(
    action: Mapping[str, Any],
    *,
    execution_id: str,
    task_id: str,
    executor: str,
    fallback_evidence_refs: list[str] | None = None,
    fallback_errors: list[str] | None = None,
) -> ExecutionResult:
    allowed_refs = list(fallback_evidence_refs or [])
    evidence_refs = _valid_refs(_string_list(action.get("evidence_refs")), allowed_refs)
    errors = [*(fallback_errors or []), *_string_list(action.get("errors"))]
    return ExecutionResult(
        execution_id=execution_id,
        task_id=task_id,
        executor=executor,
        status=_execution_status(action.get("status")),
        summary=str(action.get("summary") or ""),
        actions=_mapping_list(action.get("actions")),
        tool_results=_mapping_list(action.get("tool_results")),
        candidate_facts=_mapping_list(action.get("candidate_facts")),
        evidence_refs=evidence_refs,
        raw_output_refs=evidence_refs,
        errors=[error for error in errors if error],
        confidence=_float(action.get("confidence")),
    )


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(float(value), 1.0))
    return 0.0


def _execution_status(value: Any) -> str:
    status = str(value) if value else "failed"
    allowed = {"success", "partial", "failed", "blocked", "needs_more_information"}
    return status if status in allowed else "failed"


def _valid_refs(refs: list[str], allowed_refs: list[str]) -> list[str]:
    if not allowed_refs:
        return refs
    return [ref for ref in refs if ref in allowed_refs] or list(allowed_refs)

