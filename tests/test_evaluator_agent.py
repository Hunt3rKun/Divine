import json

import pytest

from divine.agents.evaluator import EvaluatorAgent, feedback_from_payload
from divine.blackboard import Blackboard
from divine.llm.base import LLMResponse, TokenUsage
from divine.models.common import ExecutorType, PentestPhase
from divine.models.task import TaskNode


class FakeRouter:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def chat(self, model, messages, **kwargs):
        self.calls.append((model, messages, kwargs))
        return LLMResponse(
            content=json.dumps(self.payload),
            model=model,
            usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        )


def _task():
    return TaskNode(
        id="task_001",
        description="Detect HTTP service",
        phase=PentestPhase.RECON,
        executor_type=ExecutorType.RECON,
    )


def _success_payload(evidence_refs):
    return {
        "task_id": "task_001",
        "task_judgement": {"status": "success", "completion_score": 1.0, "confidence": 0.9},
        "audit_result": {
            "confirmed_facts": [
                {
                    "type": "service",
                    "target": "http://127.0.0.1:8080",
                    "key": "protocol",
                    "value": "http",
                    "confidence": 0.9,
                    "evidence_refs": evidence_refs,
                }
            ],
            "candidate_findings": [],
            "vulnerabilities": [],
            "credentials": [],
            "sessions": [],
            "evidence_refs": evidence_refs,
            "state_updates": [],
        },
        "failure_attribution": {"level": "none", "confidence": 0.0},
        "planning_feedback": {"recommended_strategy": "expand", "should_terminate": False},
    }


def test_feedback_from_payload_filters_confirmed_facts_without_valid_evidence():
    payload = _success_payload(["missing_ref"])

    feedback = feedback_from_payload(
        payload,
        task_id="task_001",
        available_evidence_refs={"findings:http_service"},
    )

    assert feedback.task_judgement.status == "success"
    assert feedback.audit_result.evidence_refs == []
    assert feedback.audit_result.confirmed_facts == []
    assert feedback.planning_feedback.recommended_strategy == "expand"


@pytest.mark.asyncio
async def test_evaluator_audits_and_persists_feedback_to_blackboard():
    evidence_refs = ["findings:http_service"]
    blackboard = Blackboard()
    router = FakeRouter(_success_payload(evidence_refs))
    execution_result = {
        "iterations": 2,
        "finish_reason": "task_complete",
        "bb_writes": {"findings": ["http_service"]},
        "evidence_refs": evidence_refs,
    }

    feedback = await EvaluatorAgent(router, model="fake-evaluator").audit(
        task=_task(),
        execution_result=execution_result,
        blackboard=blackboard,
    )

    assert feedback.audit_result.confirmed_facts[0]["value"] == "http"
    assert blackboard.read("reflections", "audit_task_001")["task_judgement"]["status"] == "success"
    assert blackboard.read("findings", "task_001:protocol")["value"] == "http"
    assert router.calls[0][2]["agent"] == "evaluator_agent"
    assert router.calls[0][2]["trace_metadata"]["phase"] == "evaluator_audit"

