import json

import pytest

from divine.agents import EvaluatorAgent, PlannerAgent, ReconAgent
from divine.blackboard import Artifact, ExecutionResult, SharedBlackboard, TaskContext, TaskNode
from divine.llm.types import LLMResponse
from tests.test_executor_agents import FakeTools, executor_llm_for


class FakeLLMClient:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return LLMResponse(provider="fake", model="fake-evaluator", content=json.dumps(self.payload))


def success_payload(evidence_refs):
    return {
        "feedback_id": None,
        "node_id": "task_001",
        "task_judgement": {
            "status": "success",
            "completion_score": 1.0,
            "confidence": 0.9,
        },
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
        "criteria_audit": [
            {
                "criterion": "Identify service protocol",
                "met": True,
                "evidence_refs": evidence_refs,
                "reason": "HTTP response evidence supports protocol identification.",
            }
        ],
        "failure_attribution": {
            "level": "none",
            "primary_cause": None,
            "secondary_causes": [],
            "confidence": 0.0,
            "reason": None,
        },
        "planning_feedback": {
            "recommended_strategy": "expand",
            "next_focus": "Expand web testing tasks.",
            "invalidated_hypothesis": None,
            "should_terminate": False,
        },
        "needs_more_information": [],
    }


def make_blackboard():
    return SharedBlackboard(
        context=TaskContext(
            task_id="pentest_001",
            goal="Validate evaluator flow",
            target="http://127.0.0.1:8080",
            scope=["http://127.0.0.1:8080"],
        )
    )


def service_node():
    return TaskNode(
        node_id="task_001",
        task_type="service_detection",
        description="Detect HTTP service",
        target="http://127.0.0.1:8080",
        success_criteria=[
            "Identify service protocol",
            "Produce auditable evidence",
        ],
        assigned_executor="recon_agent",
    )


def recon_executor_llm():
    return executor_llm_for(
        tool_name="http_probe",
        tool_input='{"url": "http://127.0.0.1:8080"}',
        candidate_facts="""[
    {
      "type": "service",
      "target": "http://127.0.0.1:8080",
      "key": "protocol",
      "value": "http",
      "confidence": 0.9,
      "evidence_refs": ["artifact_001"],
      "reason": "HTTP probe returned a response."
    }
  ]""",
    )


class PlannerLLMClient:
    def generate(self, request):
        return LLMResponse(
            provider="fake",
            model="fake-planner",
            content="""{
  "schema_version": "planner.v1",
  "agent_name": "planner_agent",
  "status": "evolved",
  "strategy": "expand",
  "rationale": "Confirmed HTTP service supports web follow-up tasks.",
  "dag_operations": [
    {
      "operation": "create_node",
      "local_ref": "web_fingerprint",
      "node": {
        "task_type": "web_fingerprint",
        "description": "Identify Web technology and response characteristics.",
        "dependencies": ["task_001"],
        "edge_type": "dependency",
        "risk_level": "low",
        "success_criteria": ["Collect response status, headers, and title."],
        "assigned_executor": "web_agent",
        "target": "http://127.0.0.1:8080"
      },
      "reason": "HTTP service was confirmed."
    },
    {
      "operation": "create_node",
      "local_ref": "web_path_discovery",
      "node": {
        "task_type": "web_path_discovery",
        "description": "Probe common Web paths.",
        "dependencies": ["task_001"],
        "edge_type": "dependency",
        "risk_level": "low",
        "success_criteria": ["Record status codes and evidence for discovered paths."],
        "assigned_executor": "web_agent",
        "target": "http://127.0.0.1:8080"
      },
      "reason": "HTTP service was confirmed."
    }
  ],
  "needs_more_information": [],
  "termination_decision": {"should_terminate": false, "reason": null}
}""",
        )


def test_evaluator_writes_audit_feedback_and_confirms_executor_candidate_facts():
    blackboard = make_blackboard()
    node = service_node()
    blackboard.add_node(node)
    execution = ReconAgent(FakeTools(), llm_client=recon_executor_llm()).execute(node, blackboard)
    llm = FakeLLMClient(success_payload(execution.evidence_refs))

    feedback = EvaluatorAgent(llm).audit(node=node, execution_result=execution, blackboard=blackboard)

    assert feedback in blackboard.audit_feedback
    assert feedback.task_judgement.status == "success"
    assert feedback.failure_attribution.level == "none"
    assert feedback.planning_feedback.recommended_strategy == "expand"
    assert blackboard.intelligence.confirmed_facts[0].value == "http"
    assert llm.requests
    assert "Detect HTTP service" in llm.requests[0].normalized_messages()[0]["content"]
    assert execution.evidence_refs[0] in llm.requests[0].normalized_messages()[0]["content"]
    assert "nginx" in llm.requests[0].normalized_messages()[0]["content"]


def test_evaluator_feedback_can_drive_planner_without_planner_writing_audit_store():
    blackboard = make_blackboard()
    node = service_node()
    blackboard.add_node(node)
    execution = ReconAgent(FakeTools(), llm_client=recon_executor_llm()).execute(node, blackboard)
    feedback = EvaluatorAgent(FakeLLMClient(success_payload(execution.evidence_refs))).audit(
        node=node,
        execution_result=execution,
        blackboard=blackboard,
    )
    audit_count = len(blackboard.audit_feedback)

    result = PlannerAgent(PlannerLLMClient()).evolve_dag(feedback, blackboard)

    assert result.strategy == "expand"
    assert len(blackboard.audit_feedback) == audit_count
    assert {blackboard.graph.nodes[node_id].task_type for node_id in result.added_nodes} == {
        "web_fingerprint",
        "web_path_discovery",
    }


def test_evaluator_maps_missing_evidence_to_insufficient_evidence_and_replan():
    blackboard = make_blackboard()
    node = service_node()
    blackboard.add_node(node)
    execution = ExecutionResult(
        execution_id="exec_001",
        node_id=node.node_id,
        executor="recon_agent",
        status="success",
        summary="Executor claimed success without evidence.",
        candidate_facts=[
            {
                "type": "service",
                "target": "http://127.0.0.1:8080",
                "key": "protocol",
                "value": "http",
            }
        ],
        confidence=0.8,
    )
    llm = FakeLLMClient(
        {
            "node_id": "task_001",
            "task_judgement": {"status": "uncertain", "completion_score": 0.0, "confidence": 0.3},
            "audit_result": {
                "confirmed_facts": [
                    {
                        "type": "service",
                        "target": "http://127.0.0.1:8080",
                        "key": "protocol",
                        "value": "http",
                        "confidence": 0.9,
                        "evidence_refs": [],
                    }
                ],
                "candidate_findings": [],
                "vulnerabilities": [],
                "credentials": [],
                "sessions": [],
                "evidence_refs": [],
                "state_updates": [],
            },
            "failure_attribution": {
                "level": "insufficient_evidence",
                "primary_cause": "missing_evidence",
                "secondary_causes": [],
                "confidence": 0.8,
                "reason": "No evidence references were provided.",
            },
            "planning_feedback": {
                "recommended_strategy": "replan_branch",
                "next_focus": "Add validation evidence.",
                "invalidated_hypothesis": None,
                "should_terminate": False,
            },
        }
    )

    feedback = EvaluatorAgent(llm).audit(node=node, execution_result=execution, blackboard=blackboard)

    assert feedback.task_judgement.status == "uncertain"
    assert feedback.failure_attribution.level == "insufficient_evidence"
    assert feedback.planning_feedback.recommended_strategy == "replan_branch"
    assert blackboard.intelligence.confirmed_facts == []


def test_evaluator_maps_executor_errors_to_regenerate_node():
    blackboard = make_blackboard()
    node = service_node()
    blackboard.add_node(node)
    artifact_id = blackboard.next_id("artifact")
    blackboard.add_artifact(
        Artifact(
            artifact_id=artifact_id,
            artifact_type="http_response",
            source="http_probe",
            node_id=node.node_id,
            content={"status": "failed", "error": "timeout", "tool_name": "http_probe"},
        )
    )
    execution = ExecutionResult(
        execution_id="exec_001",
        node_id=node.node_id,
        executor="recon_agent",
        status="failed",
        summary="HTTP probe timed out.",
        evidence_refs=[artifact_id],
        errors=["timeout"],
        confidence=0.2,
    )
    llm = FakeLLMClient(
        {
            "node_id": "task_001",
            "task_judgement": {"status": "failed", "completion_score": 0.0, "confidence": 0.8},
            "audit_result": {
                "confirmed_facts": [],
                "candidate_findings": [],
                "vulnerabilities": [],
                "credentials": [],
                "sessions": [],
                "evidence_refs": [artifact_id],
                "state_updates": [],
            },
            "failure_attribution": {
                "level": "execution_failure",
                "primary_cause": "tool_timeout",
                "secondary_causes": ["timeout"],
                "confidence": 0.8,
                "reason": "HTTP probe timed out.",
            },
            "planning_feedback": {
                "recommended_strategy": "regenerate_node",
                "next_focus": "Retry with alternative probe settings.",
                "invalidated_hypothesis": None,
                "should_terminate": False,
            },
        }
    )

    feedback = EvaluatorAgent(llm).audit(node=node, execution_result=execution, blackboard=blackboard)

    assert feedback.task_judgement.status == "failed"
    assert feedback.failure_attribution.level == "execution_failure"
    assert feedback.planning_feedback.recommended_strategy == "regenerate_node"


def test_evaluator_rejects_llm_feedback_for_different_node_id():
    blackboard = make_blackboard()
    node = service_node()
    blackboard.add_node(node)
    execution = ReconAgent(FakeTools(), llm_client=recon_executor_llm()).execute(node, blackboard)
    payload = success_payload(execution.evidence_refs)
    payload["node_id"] = "task_999"

    with pytest.raises(ValueError, match="does not match node_id=task_001"):
        EvaluatorAgent(FakeLLMClient(payload)).audit(
            node=node,
            execution_result=execution,
            blackboard=blackboard,
        )

    assert blackboard.audit_feedback == []
