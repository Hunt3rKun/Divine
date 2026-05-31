import json

from divine.agents import EvaluatorAgent
from divine.blackboard import PlannerResult, TaskContext, TaskNode
from divine.orchestrator import Orchestrator
from divine.llm.types import LLMResponse
from tests.test_executor_agents import FakeTools


class SequencedAuditLLM:
    def __init__(self):
        self.requests = []
        self.executor_calls = {}

    def generate(self, request):
        self.requests.append(request)
        if request.agent == "planner_agent":
            content = request.normalized_messages()[0]["content"]
            if "initial DAG" in content:
                return LLMResponse(provider="fake", model="fake-planner", content=_planner_initial_payload())
            if "technology" in content:
                return LLMResponse(provider="fake", model="fake-planner", content=_planner_web_rule_payload())
            return LLMResponse(provider="fake", model="fake-planner", content=_planner_service_expand_payload())
        if request.agent in {"recon_agent", "web_agent", "host_agent"}:
            content = request.normalized_messages()[0]["content"]
            node_id = _current_node_id(content)
            count = self.executor_calls.get(node_id, 0) + 1
            self.executor_calls[node_id] = count
            if count == 1:
                if "web_path_discovery" in content:
                    return LLMResponse(provider="fake", model="fake-executor", content=_executor_tool_call("path_probe", '{"base_url": "http://127.0.0.1:8080"}'))
                return LLMResponse(provider="fake", model="fake-executor", content=_executor_tool_call("http_probe", '{"url": "http://127.0.0.1:8080"}'))
            return LLMResponse(provider="fake", model="fake-executor", content=_executor_final_payload(node_id))
        content = request.normalized_messages()[0]["content"]
        node_id = _current_node_id(content)
        if "service_detection" in content:
            artifact_id = _first_artifact_ref(content)
            payload = _success_payload(
                node_id=node_id,
                artifact_id=artifact_id,
                fact_type="service",
                key="protocol",
                value="http",
                target="http://127.0.0.1:8080",
                next_focus="Expand web testing tasks.",
            )
        else:
            artifact_id = _first_artifact_ref(content)
            payload = _success_payload(
                node_id=node_id,
                artifact_id=artifact_id,
                fact_type="technology",
                key="server",
                value="nginx",
                target="http://127.0.0.1:8080",
                next_focus="Expand technology rule checks.",
            )
        return LLMResponse(provider="fake", model="fake-evaluator", content=json.dumps(payload))


def test_orchestrator_runs_planner_executor_evaluator_planner_loop_once():
    context = TaskContext(
        task_id="pentest_001",
        goal="Validate full loop",
        target="http://127.0.0.1:8080",
        scope=["http://127.0.0.1:8080"],
        max_iterations=1,
    )
    llm = SequencedAuditLLM()

    result = Orchestrator(evaluator=EvaluatorAgent(llm), llm_client=llm, tools=FakeTools()).run(context)

    blackboard = result.blackboard
    assert result.iterations == 1
    assert result.stop_reason == "max_iterations"
    assert len(blackboard.execution_results) == 1
    assert len(blackboard.audit_feedback) == 1
    assert len([request for request in llm.requests if request.agent == "evaluator_agent"]) == 1
    assert any(request.agent == "planner_agent" for request in llm.requests)
    assert any(request.agent == "recon_agent" for request in llm.requests)
    assert blackboard.intelligence.confirmed_facts[0].value == "http"
    assert {node.task_type for node in blackboard.graph.nodes.values()} == {
        "service_detection",
        "web_fingerprint",
        "web_path_discovery",
    }
    assert [event.event_type for event in blackboard.event_log].count("DAGEvolved") == 2


def test_orchestrator_continues_to_next_expanded_node():
    context = TaskContext(
        task_id="pentest_001",
        goal="Validate two loop iterations",
        target="http://127.0.0.1:8080",
        scope=["http://127.0.0.1:8080"],
        max_iterations=2,
    )
    llm = SequencedAuditLLM()

    result = Orchestrator(evaluator=EvaluatorAgent(llm), llm_client=llm, tools=FakeTools()).run(context)

    blackboard = result.blackboard
    assert result.iterations == 2
    assert result.stop_reason == "max_iterations"
    assert len(blackboard.execution_results) == 2
    assert len(blackboard.audit_feedback) == 2
    assert len([request for request in llm.requests if request.agent == "evaluator_agent"]) == 2
    assert any(request.agent == "planner_agent" for request in llm.requests)
    assert any(request.agent == "web_agent" for request in llm.requests)
    assert any(node.task_type == "web_rule_check" for node in blackboard.graph.nodes.values())


def test_orchestrator_stops_when_task_type_is_unsupported():
    context = TaskContext(
        task_id="pentest_001",
        goal="Validate route blocked",
        target="http://127.0.0.1:8080",
        scope=["http://127.0.0.1:8080"],
    )

    class UnsupportedTaskPlanner:
        def generate_initial_dag(self, context, blackboard):
            blackboard.add_node(
                TaskNode(
                    node_id="task_001",
                    task_type="unsupported_task",
                    description="Unsupported task",
                )
            )
            return PlannerResult(
                status="planned",
                strategy="initial_plan",
                rationale="test",
            )

        def evolve_dag(self, feedback, blackboard):
            raise AssertionError("planner should not evolve after route block")

    result = Orchestrator(
        planner=UnsupportedTaskPlanner(),
        evaluator=EvaluatorAgent(SequencedAuditLLM()),
        llm_client=SequencedAuditLLM(),
        tools=FakeTools(),
    ).run(context)

    assert result.stop_reason == "route_blocked"
    assert result.blackboard.graph.nodes["task_001"].status == "blocked"
    assert result.blackboard.execution_results == []


def test_orchestrator_records_initial_planner_failure_instead_of_raising():
    context = TaskContext(
        task_id="pentest_001",
        goal="Validate initial planner failure handling",
        target="http://127.0.0.1:8080",
        scope=["http://127.0.0.1:8080"],
    )

    class BrokenInitialPlanner:
        def generate_initial_dag(self, context, blackboard):
            raise RuntimeError("initial planner exploded")

        def evolve_dag(self, feedback, blackboard):
            raise AssertionError("planner should not evolve")

    result = Orchestrator(
        planner=BrokenInitialPlanner(),
        evaluator=EvaluatorAgent(SequencedAuditLLM()),
        llm_client=SequencedAuditLLM(),
        tools=FakeTools(),
    ).run(context)

    assert result.stop_reason == "planner_failed"
    assert result.iterations == 0
    assert result.blackboard.event_log[-1].event_type == "PlannerFailed"
    assert "initial planner exploded" in result.blackboard.event_log[-1].payload["error"]


def test_orchestrator_records_empty_initial_dag_as_planner_failure():
    context = TaskContext(
        task_id="pentest_001",
        goal="Validate empty initial DAG handling",
        target="http://127.0.0.1:8080",
        scope=["http://127.0.0.1:8080"],
    )

    class EmptyPlanner:
        def generate_initial_dag(self, context, blackboard):
            return PlannerResult(status="planned", strategy="initial_plan", rationale="empty")

        def evolve_dag(self, feedback, blackboard):
            raise AssertionError("planner should not evolve")

    result = Orchestrator(
        planner=EmptyPlanner(),
        evaluator=EvaluatorAgent(SequencedAuditLLM()),
        llm_client=SequencedAuditLLM(),
        tools=FakeTools(),
    ).run(context)

    assert result.stop_reason == "planner_failed"
    assert result.iterations == 0
    assert result.blackboard.event_log[-1].event_type == "PlannerFailed"
    assert result.blackboard.event_log[-1].message == "Initial planner produced an empty DAG"


def test_orchestrator_records_executor_failure_instead_of_raising():
    context = TaskContext(
        task_id="pentest_001",
        goal="Validate executor failure handling",
        target="http://127.0.0.1:8080",
        scope=["http://127.0.0.1:8080"],
    )

    class OneNodePlanner:
        def generate_initial_dag(self, context, blackboard):
            blackboard.add_node(
                TaskNode(
                    node_id="task_001",
                    task_type="service_detection",
                    description="Detect service",
                    target=context.target,
                    success_criteria=["Identify service protocol"],
                    assigned_executor="recon_agent",
                )
            )
            return PlannerResult(status="planned", strategy="initial_plan", rationale="test", added_nodes=["task_001"])

        def evolve_dag(self, feedback, blackboard):
            raise AssertionError("planner should not evolve")

    class BrokenExecutor:
        def execute(self, node, blackboard):
            raise RuntimeError("executor exploded")

    class UnusedEvaluator:
        def audit(self, *, node, execution_result, blackboard):
            raise AssertionError("evaluator should not run")

    result = Orchestrator(
        planner=OneNodePlanner(),
        evaluator=UnusedEvaluator(),
        executors={"recon_agent": BrokenExecutor()},
    ).run(context)

    assert result.stop_reason == "executor_failed"
    assert result.iterations == 1
    assert result.blackboard.graph.nodes["task_001"].status == "blocked"
    assert result.blackboard.event_log[-1].event_type == "ExecutionFailed"
    assert "executor exploded" in result.blackboard.event_log[-1].payload["error"]


def test_orchestrator_stops_after_max_consecutive_failures():
    context = TaskContext(
        task_id="pentest_001",
        goal="Validate failure stop",
        target="http://127.0.0.1:8080",
        scope=["http://127.0.0.1:8080"],
        max_iterations=3,
        max_consecutive_failures=1,
    )

    class FailingAuditLLM:
        def generate(self, request):
            artifact_id = _first_artifact_ref(request.normalized_messages()[0]["content"])
            node_id = _current_node_id(request.normalized_messages()[0]["content"])
            return LLMResponse(
                provider="fake",
                model="fake-evaluator",
                content=json.dumps(
                    {
                        "node_id": node_id,
                        "task_judgement": {
                            "status": "failed",
                            "completion_score": 0.0,
                            "confidence": 0.8,
                        },
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
                            "primary_cause": "forced_failure",
                            "secondary_causes": [],
                            "confidence": 0.8,
                            "reason": "forced failure",
                        },
                        "planning_feedback": {
                            "recommended_strategy": "regenerate_node",
                            "next_focus": "retry",
                            "invalidated_hypothesis": None,
                            "should_terminate": False,
                        },
                    }
                ),
            )

    result = Orchestrator(evaluator=EvaluatorAgent(FailingAuditLLM()), llm_client=SequencedAuditLLM(), tools=FakeTools()).run(context)

    assert result.stop_reason == "max_consecutive_failures"
    assert result.iterations == 1
    assert len(result.blackboard.audit_feedback) == 1


def test_orchestrator_records_evaluator_failure_instead_of_raising():
    context = TaskContext(
        task_id="pentest_001",
        goal="Validate evaluator failure handling",
        target="http://127.0.0.1:8080",
        scope=["http://127.0.0.1:8080"],
        max_iterations=1,
    )

    class BrokenLLM:
        def generate(self, request):
            return LLMResponse(provider="fake", model="fake-evaluator", content="not json")

    result = Orchestrator(evaluator=EvaluatorAgent(BrokenLLM()), llm_client=SequencedAuditLLM(), tools=FakeTools()).run(context)

    assert result.stop_reason == "evaluator_failed"
    assert result.iterations == 1
    assert result.blackboard.graph.nodes["task_001"].status == "blocked"
    assert result.blackboard.event_log[-1].event_type == "AuditFailed"


def test_orchestrator_records_planner_failure_instead_of_raising():
    context = TaskContext(
        task_id="pentest_001",
        goal="Validate planner failure handling",
        target="http://127.0.0.1:8080",
        scope=["http://127.0.0.1:8080"],
        max_iterations=1,
    )

    class BrokenPlanner:
        def generate_initial_dag(self, context, blackboard):
            blackboard.add_node(
                TaskNode(
                    node_id="task_001",
                    task_type="service_detection",
                    description="Detect service",
                    target=context.target,
                    success_criteria=["Identify service protocol"],
                )
            )
            return PlannerResult(status="planned", strategy="initial_plan", rationale="test")

        def evolve_dag(self, feedback, blackboard):
            raise RuntimeError("planner exploded")

    result = Orchestrator(
        planner=BrokenPlanner(),
        evaluator=EvaluatorAgent(SequencedAuditLLM()),
        llm_client=SequencedAuditLLM(),
        tools=FakeTools(),
    ).run(context)

    assert result.stop_reason == "planner_failed"
    assert result.iterations == 1
    assert result.blackboard.graph.nodes["task_001"].status == "blocked"
    assert result.blackboard.event_log[-1].event_type == "PlannerFailed"


def _success_payload(*, node_id, artifact_id, fact_type, key, value, target, next_focus):
    return {
        "feedback_id": None,
        "node_id": node_id,
        "task_judgement": {"status": "success", "completion_score": 1.0, "confidence": 0.9},
        "audit_result": {
            "confirmed_facts": [
                {
                    "type": fact_type,
                    "target": target,
                    "key": key,
                    "value": value,
                    "confidence": 0.9,
                    "evidence_refs": [artifact_id],
                }
            ],
            "candidate_findings": [],
            "vulnerabilities": [],
            "credentials": [],
            "sessions": [],
            "evidence_refs": [artifact_id],
            "state_updates": [],
        },
        "criteria_audit": [],
        "failure_attribution": {
            "level": "none",
            "primary_cause": None,
            "secondary_causes": [],
            "confidence": 0.0,
            "reason": None,
        },
        "planning_feedback": {
            "recommended_strategy": "expand",
            "next_focus": next_focus,
            "invalidated_hypothesis": None,
            "should_terminate": False,
        },
        "needs_more_information": [],
    }


def _planner_initial_payload():
    return """{
  "schema_version": "planner.v1",
  "agent_name": "planner_agent",
  "status": "planned",
  "strategy": "initial_plan",
  "rationale": "Create initial service detection.",
  "dag_operations": [
    {"operation": "create_node", "local_ref": "service", "node": {"task_type": "service_detection", "description": "Detect service.", "dependencies": [], "edge_type": "dependency", "risk_level": "low", "success_criteria": ["Identify service protocol"], "assigned_executor": "recon_agent", "target": "http://127.0.0.1:8080"}, "reason": "Start with service detection."}
  ],
  "needs_more_information": [],
  "termination_decision": {"should_terminate": false, "reason": null}
}"""


def _planner_service_expand_payload():
    return """{
  "schema_version": "planner.v1",
  "agent_name": "planner_agent",
  "status": "evolved",
  "strategy": "expand",
  "rationale": "Expand web work.",
  "dag_operations": [
    {"operation": "create_node", "local_ref": "fingerprint", "node": {"task_type": "web_fingerprint", "description": "Identify web stack.", "dependencies": ["task_001"], "edge_type": "dependency", "risk_level": "low", "success_criteria": ["Collect response headers and title."], "assigned_executor": "web_agent", "target": "http://127.0.0.1:8080"}, "reason": "HTTP service confirmed."},
    {"operation": "create_node", "local_ref": "paths", "node": {"task_type": "web_path_discovery", "description": "Discover web paths.", "dependencies": ["task_001"], "edge_type": "dependency", "risk_level": "low", "success_criteria": ["Record path status codes."], "assigned_executor": "web_agent", "target": "http://127.0.0.1:8080"}, "reason": "HTTP service confirmed."}
  ],
  "needs_more_information": [],
  "termination_decision": {"should_terminate": false, "reason": null}
}"""


def _planner_web_rule_payload():
    return """{
  "schema_version": "planner.v1",
  "agent_name": "planner_agent",
  "status": "evolved",
  "strategy": "expand",
  "rationale": "Add web rule checks.",
  "dag_operations": [
    {"operation": "create_node", "local_ref": "rules", "node": {"task_type": "web_rule_check", "description": "Run web rule checks.", "dependencies": ["task_002"], "edge_type": "dependency", "risk_level": "low", "success_criteria": ["Tie findings to evidence."], "assigned_executor": "web_agent", "target": "http://127.0.0.1:8080"}, "reason": "Technology evidence was confirmed."}
  ],
  "needs_more_information": [],
  "termination_decision": {"should_terminate": false, "reason": null}
}"""


def _executor_tool_call(tool_name, tool_input):
    return f"""{{
  "schema_version": "executor.v1",
  "agent_name": "executor_agent",
  "action_type": "tool_call",
  "node_id": "task_001",
  "tool_name": "{tool_name}",
  "tool_input": {tool_input},
  "rationale": "Collect evidence."
}}"""


def _executor_final_payload(node_id):
    return f"""{{
  "schema_version": "executor.v1",
  "agent_name": "executor_agent",
  "action_type": "final_result",
  "node_id": "{node_id}",
  "status": "success",
  "summary": "Evidence collected.",
  "actions": [{{"tool_name": "http_probe", "tool_input": {{"url": "http://127.0.0.1:8080"}}, "output_ref": "artifact_001", "status": "success", "error": null}}],
  "candidate_facts": [],
  "evidence_refs": ["artifact_001"],
  "errors": [],
  "confidence": 0.9
}}"""


def _first_artifact_ref(content):
    marker = "artifact_"
    index = content.find(marker)
    assert index >= 0, content
    return content[index : index + len("artifact_001")]


def _current_node_id(content):
    marker = '"node_id": "'
    index = content.find(marker)
    assert index >= 0, content
    start = index + len(marker)
    end = content.find('"', start)
    return content[start:end]
