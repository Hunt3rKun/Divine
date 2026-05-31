import pytest

from divine.agents import PlannerAgent
from divine.blackboard import (
    AuditFeedback,
    AuditResult,
    FailureAttribution,
    PlanningFeedback,
    SharedBlackboard,
    TaskContext,
    TaskEdge,
    TaskJudgement,
    TaskNode,
)
from divine.llm.types import LLMResponse


def make_blackboard() -> SharedBlackboard:
    return SharedBlackboard(
        context=TaskContext(
            task_id="pentest_001",
            goal="Validate framework flow against a local target",
            target="http://127.0.0.1:8080",
            scope=["127.0.0.1:8080"],
        )
    )


def feedback(
    *,
    node_id: str,
    status: str = "success",
    confirmed_facts=None,
    failure_level: str = "none",
    reason: str | None = None,
    recommended_strategy: str | None = None,
    invalidated_hypothesis: str | None = None,
) -> AuditFeedback:
    return AuditFeedback(
        feedback_id="fb_001",
        node_id=node_id,
        task_judgement=TaskJudgement(status=status, completion_score=0.9, confidence=0.9),
        audit_result=AuditResult(
            confirmed_facts=list(confirmed_facts or []),
            evidence_refs=["artifact_001"],
        ),
        failure_attribution=FailureAttribution(
            level=failure_level,
            primary_cause=reason,
            confidence=0.8 if failure_level != "none" else 0.0,
            reason=reason,
        ),
        planning_feedback=PlanningFeedback(
            recommended_strategy=recommended_strategy,
            invalidated_hypothesis=invalidated_hypothesis,
        ),
    )


class SequencedPlannerLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate(self, request):
        return LLMResponse(provider="fake", model="fake-planner", content=self.responses.pop(0))


def initial_plan_payload(*, risk_level="low"):
    return f"""{{
  "schema_version": "planner.v1",
  "agent_name": "planner_agent",
  "status": "planned",
  "strategy": "initial_plan",
  "rationale": "Create the first node.",
  "dag_operations": [
    {{
      "operation": "create_node",
      "local_ref": "initial_service_detection",
      "node": {{
        "task_type": "service_detection",
        "description": "Identify reachable services for the target.",
        "dependencies": [],
        "edge_type": "dependency",
        "risk_level": "{risk_level}",
        "success_criteria": ["Collect service evidence."],
        "assigned_executor": "recon_agent",
        "target": "http://127.0.0.1:8080"
      }},
      "reason": "The task needs an initial service baseline."
    }}
  ],
  "needs_more_information": [],
  "termination_decision": {{"should_terminate": false, "reason": null}}
}}"""


def service_expand_payload():
    return """{
  "schema_version": "planner.v1",
  "agent_name": "planner_agent",
  "status": "evolved",
  "strategy": "expand",
  "rationale": "Confirmed HTTP service supports web tasks.",
  "dag_operations": [
    {
      "operation": "update_node_status",
      "node_id": "task_001",
      "status": "completed",
      "reason": "Service detection result was evaluated."
    },
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
}"""


def test_generate_initial_dag_creates_low_risk_service_detection_node():
    blackboard = make_blackboard()
    result = PlannerAgent(SequencedPlannerLLM([initial_plan_payload()])).generate_initial_dag(blackboard.context, blackboard)

    assert result.status == "planned"
    assert result.strategy == "initial_plan"
    assert result.added_nodes == ["task_001"]
    assert blackboard.graph.nodes["task_001"].task_type == "service_detection"
    assert blackboard.graph.nodes["task_001"].assigned_executor == "recon_agent"
    assert blackboard.event_log[-1].payload["strategy"] == "initial_plan"
    assert not hasattr(blackboard, "dag_versions")


def test_llm_planner_applies_create_node_with_local_ref():
    blackboard = make_blackboard()

    class FakePlannerLLM:
        def generate(self, request):
            return LLMResponse(
                provider="fake",
                model="fake-planner",
                content="""{
  "schema_version": "planner.v1",
  "agent_name": "planner_agent",
  "status": "planned",
  "strategy": "initial_plan",
  "rationale": "Create the first node.",
  "dag_operations": [
    {
      "operation": "create_node",
      "local_ref": "initial_service_detection",
      "node": {
        "task_type": "service_detection",
        "description": "Identify reachable services for the target.",
        "dependencies": [],
        "edge_type": "dependency",
        "risk_level": "medium",
        "success_criteria": ["Collect service evidence."],
        "assigned_executor": "recon_agent",
        "target": "http://127.0.0.1:8080"
      },
      "reason": "The task needs an initial service baseline."
    }
  ],
  "needs_more_information": [],
  "termination_decision": {"should_terminate": false, "reason": null}
}""",
            )

    result = PlannerAgent(FakePlannerLLM()).generate_initial_dag(blackboard.context, blackboard)

    assert result.operations[0].operation == "create_node"
    assert result.added_nodes == ["task_001"]
    assert blackboard.graph.nodes["task_001"].task_type == "service_detection"
    assert blackboard.graph.nodes["task_001"].risk_level == "medium"


def test_llm_planner_rejects_missing_dependencies_before_writing_node():
    blackboard = make_blackboard()
    payload = """{
  "schema_version": "planner.v1",
  "agent_name": "planner_agent",
  "status": "planned",
  "strategy": "initial_plan",
  "rationale": "Invalid dependency.",
  "dag_operations": [
    {
      "operation": "create_node",
      "local_ref": "bad_node",
      "node": {
        "task_type": "web_fingerprint",
        "description": "Invalid node.",
        "dependencies": ["task_missing"],
        "edge_type": "dependency",
        "risk_level": "low",
        "success_criteria": ["Collect evidence."],
        "assigned_executor": "web_agent",
        "target": "http://127.0.0.1:8080"
      },
      "reason": "Missing dependency should be rejected."
    }
  ],
  "needs_more_information": [],
  "termination_decision": {"should_terminate": false, "reason": null}
}"""

    with pytest.raises(ValueError, match="dependencies are missing"):
        PlannerAgent(SequencedPlannerLLM([payload])).generate_initial_dag(blackboard.context, blackboard)

    assert blackboard.graph.nodes == {}


def test_llm_planner_rejects_unsupported_executor_before_writing_node():
    blackboard = make_blackboard()
    payload = """{
  "schema_version": "planner.v1",
  "agent_name": "planner_agent",
  "status": "planned",
  "strategy": "initial_plan",
  "rationale": "Invalid executor.",
  "dag_operations": [
    {
      "operation": "create_node",
      "local_ref": "bad_node",
      "node": {
        "task_type": "service_detection",
        "description": "Invalid node.",
        "dependencies": [],
        "edge_type": "dependency",
        "risk_level": "low",
        "success_criteria": ["Collect evidence."],
        "assigned_executor": "report_agent",
        "target": "http://127.0.0.1:8080"
      },
      "reason": "Unsupported executor should be rejected."
    }
  ],
  "needs_more_information": [],
  "termination_decision": {"should_terminate": false, "reason": null}
}"""

    with pytest.raises(ValueError, match="Unsupported planner assigned_executor"):
        PlannerAgent(SequencedPlannerLLM([payload])).generate_initial_dag(blackboard.context, blackboard)

    assert blackboard.graph.nodes == {}


def test_llm_planner_rejects_added_nodes_depending_on_partial_node():
    blackboard = make_blackboard()
    trigger = TaskNode(
        "task_001",
        "service_detection",
        "Detect service",
        status="partial",
        target=blackboard.context.target,
    )
    blackboard.add_node(trigger)
    audit_feedback = feedback(
        node_id="task_001",
        status="partial",
        failure_level="insufficient_evidence",
        reason="Need more evidence.",
    )
    blackboard.add_audit_feedback(audit_feedback)
    payload = """{
  "schema_version": "planner.v1",
  "agent_name": "planner_agent",
  "status": "evolved",
  "strategy": "expand",
  "rationale": "Invalid dependency status.",
  "dag_operations": [
    {
      "operation": "create_node",
      "local_ref": "bad_followup",
      "node": {
        "task_type": "web_fingerprint",
        "description": "Follow up from partial node.",
        "dependencies": ["task_001"],
        "edge_type": "dependency",
        "risk_level": "low",
        "success_criteria": ["Collect evidence."],
        "assigned_executor": "web_agent",
        "target": "http://127.0.0.1:8080"
      },
      "reason": "This would be stuck because task_001 is partial."
    }
  ],
  "needs_more_information": [],
  "termination_decision": {"should_terminate": false, "reason": null}
}"""

    with pytest.raises(ValueError, match="dependencies are non-executable"):
        PlannerAgent(SequencedPlannerLLM([payload])).evolve_dag(audit_feedback, blackboard)

    assert set(blackboard.graph.nodes) == {"task_001"}


def test_graph_add_dependency_edge_updates_target_dependencies():
    blackboard = make_blackboard()
    root = TaskNode("task_001", "service_detection", "Detect service")
    child = TaskNode("task_002", "web_fingerprint", "Identify web stack")
    blackboard.add_node(root)
    blackboard.add_node(child)

    blackboard.add_edge(TaskEdge("task_001", "task_002", "dependency", "service before web"))

    assert blackboard.graph.nodes["task_002"].dependencies == ["task_001"]
    assert blackboard.graph.executable_nodes() == [root]


def test_success_with_http_service_fact_expands_web_tasks_and_confirms_fact():
    blackboard = make_blackboard()
    planner = PlannerAgent(SequencedPlannerLLM([initial_plan_payload(), service_expand_payload()]))
    planner.generate_initial_dag(blackboard.context, blackboard)
    audit_feedback = feedback(
        node_id="task_001",
        confirmed_facts=[
            {
                "type": "service",
                "target": "http://127.0.0.1:8080",
                "key": "protocol",
                "value": "http",
                "confidence": 0.95,
            }
        ],
    )
    blackboard.add_audit_feedback(audit_feedback)

    result = planner.evolve_dag(
        audit_feedback,
        blackboard,
    )

    assert result.strategy == "expand"
    assert {blackboard.graph.nodes[node_id].task_type for node_id in result.added_nodes} == {
        "web_fingerprint",
        "web_path_discovery",
    }
    assert blackboard.graph.nodes["task_001"].status == "completed"
    assert blackboard.intelligence.confirmed_facts[0].value == "http"
    assert blackboard.event_log[-1].payload["added_nodes"] == result.added_nodes


def test_planner_evolve_does_not_write_audit_feedback():
    blackboard = make_blackboard()
    planner = PlannerAgent(SequencedPlannerLLM([initial_plan_payload(), service_expand_payload()]))
    planner.generate_initial_dag(blackboard.context, blackboard)
    audit_feedback = feedback(
        node_id="task_001",
        confirmed_facts=[
            {
                "type": "service",
                "target": "http://127.0.0.1:8080",
                "key": "protocol",
                "value": "http",
                "confidence": 0.95,
            }
        ],
    )
    blackboard.add_audit_feedback(audit_feedback)
    audit_count = len(blackboard.audit_feedback)

    planner.evolve_dag(audit_feedback, blackboard)

    assert len(blackboard.audit_feedback) == audit_count


def test_planner_rejects_feedback_not_written_by_evaluator_to_blackboard():
    blackboard = make_blackboard()
    planner = PlannerAgent(SequencedPlannerLLM([initial_plan_payload(), service_expand_payload()]))
    planner.generate_initial_dag(blackboard.context, blackboard)
    audit_feedback = feedback(
        node_id="task_001",
        confirmed_facts=[
            {
                "type": "service",
                "target": "http://127.0.0.1:8080",
                "key": "protocol",
                "value": "http",
                "confidence": 0.95,
            }
        ],
    )

    with pytest.raises(ValueError, match="must be written to the blackboard"):
        planner.evolve_dag(audit_feedback, blackboard)


def test_execution_failure_regenerates_alternative_node_until_retry_budget_is_exhausted():
    blackboard = make_blackboard()
    planner = PlannerAgent(SequencedPlannerLLM([initial_plan_payload(), """{
  "schema_version": "planner.v1",
  "agent_name": "planner_agent",
  "status": "evolved",
  "strategy": "regenerate_node",
  "rationale": "Retry with an alternative node.",
  "dag_operations": [
    {
      "operation": "create_node",
      "local_ref": "retry_service_detection",
      "node": {
        "task_type": "service_detection",
        "description": "Alternative attempt for service detection.",
        "dependencies": [],
        "edge_type": "alternative",
        "risk_level": "low",
        "success_criteria": ["Collect service evidence."],
        "assigned_executor": "recon_agent",
        "target": "http://127.0.0.1:8080"
      },
      "reason": "Previous execution failed."
    }
  ],
  "needs_more_information": [],
  "termination_decision": {"should_terminate": false, "reason": null}
}"""]))
    planner.generate_initial_dag(blackboard.context, blackboard)

    audit_feedback = feedback(
        node_id="task_001",
        status="failed",
        failure_level="execution_failure",
        reason="HTTP probe timed out",
    )
    blackboard.add_audit_feedback(audit_feedback)

    result = planner.evolve_dag(audit_feedback, blackboard)

    replacement = blackboard.graph.nodes[result.added_nodes[0]]
    assert result.strategy == "regenerate_node"
    assert replacement.task_type == "service_detection"
    assert replacement.edge_type == "alternative"
    assert replacement in blackboard.graph.executable_nodes()


def test_insufficient_evidence_adds_executable_validation_node():
    blackboard = make_blackboard()
    planner = PlannerAgent(SequencedPlannerLLM([initial_plan_payload(), """{
  "schema_version": "planner.v1",
  "agent_name": "planner_agent",
  "status": "evolved",
  "strategy": "replan_branch",
  "rationale": "Add validation evidence.",
  "dag_operations": [
    {
      "operation": "create_node",
      "local_ref": "service_detection_validation",
      "node": {
        "task_type": "service_detection_validation",
        "description": "Collect additional evidence to validate service detection.",
        "dependencies": [],
        "edge_type": "validation",
        "risk_level": "low",
        "success_criteria": ["Resolve the prior uncertainty."],
        "assigned_executor": "recon_agent",
        "target": "http://127.0.0.1:8080"
      },
      "reason": "Evidence was inconclusive."
    }
  ],
  "needs_more_information": [],
  "termination_decision": {"should_terminate": false, "reason": null}
}"""]))
    planner.generate_initial_dag(blackboard.context, blackboard)

    audit_feedback = feedback(
        node_id="task_001",
        status="uncertain",
        failure_level="insufficient_evidence",
        reason="HTTP probe evidence was inconclusive",
    )
    blackboard.add_audit_feedback(audit_feedback)

    result = planner.evolve_dag(audit_feedback, blackboard)

    validation = blackboard.graph.nodes[result.added_nodes[0]]
    assert result.strategy == "replan_branch"
    assert validation.task_type == "service_detection_validation"
    assert validation.dependencies == []
    assert validation in blackboard.graph.executable_nodes()


def test_cognitive_failure_replans_branch_into_decomposed_observation_tasks():
    blackboard = make_blackboard()
    planner = PlannerAgent(SequencedPlannerLLM([initial_plan_payload(), """{
  "schema_version": "planner.v1",
  "agent_name": "planner_agent",
  "status": "evolved",
  "strategy": "replan_branch",
  "rationale": "Decompose observations.",
  "dag_operations": [
    {"operation": "create_node", "local_ref": "headers", "node": {"task_type": "response_header_analysis", "description": "Analyze response headers.", "dependencies": ["task_001"], "edge_type": "alternative", "risk_level": "low", "success_criteria": ["Collect header observations."], "assigned_executor": "web_agent", "target": "http://127.0.0.1:8080"}, "reason": "Need narrower observation."},
    {"operation": "create_node", "local_ref": "title", "node": {"task_type": "page_title_extraction", "description": "Extract page title.", "dependencies": ["task_001"], "edge_type": "alternative", "risk_level": "low", "success_criteria": ["Collect title observations."], "assigned_executor": "web_agent", "target": "http://127.0.0.1:8080"}, "reason": "Need narrower observation."},
    {"operation": "create_node", "local_ref": "static", "node": {"task_type": "static_resource_analysis", "description": "Inspect static resource references.", "dependencies": ["task_001"], "edge_type": "alternative", "risk_level": "low", "success_criteria": ["Collect static resource observations."], "assigned_executor": "web_agent", "target": "http://127.0.0.1:8080"}, "reason": "Need narrower observation."}
  ],
  "needs_more_information": [],
  "termination_decision": {"should_terminate": false, "reason": null}
}"""]))
    planner.generate_initial_dag(blackboard.context, blackboard)
    blackboard.graph.nodes["task_001"].status = "success"
    node = TaskNode(
        node_id="task_002",
        task_type="web_fingerprint",
        description="Identify web stack",
        dependencies=["task_001"],
        assigned_executor="web_agent",
        target=blackboard.context.target,
    )
    blackboard.add_node(node)
    blackboard.add_edge(TaskEdge("task_001", "task_002", "dependency", "web service found"))

    audit_feedback = feedback(
        node_id="task_002",
        status="failed",
        failure_level="cognitive_failure",
        reason="Fingerprint hypothesis was too broad",
    )
    blackboard.add_audit_feedback(audit_feedback)

    result = planner.evolve_dag(audit_feedback, blackboard)

    assert result.strategy == "replan_branch"
    assert {blackboard.graph.nodes[node_id].task_type for node_id in result.added_nodes} == {
        "response_header_analysis",
        "page_title_extraction",
        "static_resource_analysis",
    }
    assert all(blackboard.graph.nodes[node_id].dependencies == ["task_001"] for node_id in result.added_nodes)


def test_strategic_failure_replans_branch_and_prunes_downstream_without_deleting_nodes():
    blackboard = make_blackboard()
    planner = PlannerAgent(SequencedPlannerLLM(["""{
  "schema_version": "planner.v1",
  "agent_name": "planner_agent",
  "status": "evolved",
  "strategy": "replan_branch",
  "rationale": "Prune invalid downstream nodes.",
  "dag_operations": [
    {"operation": "update_node_status", "node_id": "task_002", "status": "pruned", "reason": "Branch invalidated."},
    {"operation": "update_node_status", "node_id": "task_003", "status": "pruned", "reason": "Branch invalidated."}
  ],
  "needs_more_information": [],
  "termination_decision": {"should_terminate": false, "reason": null}
}"""]))
    root = TaskNode("task_001", "service_detection", "Detect service")
    child = TaskNode("task_002", "web_fingerprint", "Identify web stack", dependencies=["task_001"])
    grandchild = TaskNode("task_003", "web_rule_check", "Check web rules", dependencies=["task_002"])
    blackboard.add_node(root)
    blackboard.add_node(child)
    blackboard.add_node(grandchild)
    blackboard.add_edge(TaskEdge("task_001", "task_002", "hypothesis", "web service exists"))
    blackboard.add_edge(TaskEdge("task_002", "task_003", "dependency", "technology identified"))

    audit_feedback = feedback(
        node_id="task_001",
        status="failed",
        failure_level="strategic_failure",
        reason="Evidence indicates no web service is present",
        invalidated_hypothesis="target exposes a web service",
    )
    blackboard.add_audit_feedback(audit_feedback)

    result = planner.evolve_dag(audit_feedback, blackboard)

    assert result.strategy == "replan_branch"
    assert result.pruned_nodes == ["task_002", "task_003"]
    assert set(blackboard.graph.nodes) == {"task_001", "task_002", "task_003"}
    assert blackboard.graph.nodes["task_002"].status == "pruned"
