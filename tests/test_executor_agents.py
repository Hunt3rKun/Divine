from divine.agents import ExecutionRouter, HostAgent, ReconAgent, WebAgent
from divine.blackboard import SharedBlackboard, TaskContext, TaskNode
from divine.llm.types import LLMResponse
from divine.tools import ToolResult


class FakeTools:
    def tcp_connect_check(self, host, port, *, timeout=2.0):
        return ToolResult(
            "tcp_connect_check",
            "success",
            {"host": host, "port": port, "timeout": timeout},
            {"reachable": True, "host": host, "port": port},
            artifact_type="tcp_connect",
        )

    def http_probe(self, url, *, timeout=5.0):
        return ToolResult(
            "http_probe",
            "success",
            {"url": url, "timeout": timeout},
            {
                "url": url,
                "status_code": 200,
                "headers": {"server": "nginx", "x-powered-by": "Python"},
                "title": "Local Lab",
                "body_preview": "<title>Local Lab</title>",
            },
            artifact_type="http_response",
        )

    def https_probe(self, url, *, timeout=5.0):
        return self.http_probe(url, timeout=timeout)

    def path_probe(self, base_url, paths=None, *, timeout=5.0):
        return ToolResult(
            "path_probe",
            "success",
            {"base_url": base_url, "paths": paths or ["/"], "timeout": timeout},
            {
                "results": [{"path": "/", "url": base_url, "status": "success", "status_code": 200}],
                "discovered": [{"path": "/", "url": base_url, "status": "success", "status_code": 200}],
            },
            artifact_type="web_path_probe",
        )

    def host_info(self):
        return ToolResult(
            "host_info",
            "success",
            {},
            {"platform": "Linux-test", "system": "Linux"},
            artifact_type="host_info",
        )


class SequencedExecutorLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        return LLMResponse(
            provider="fake",
            model="fake-executor",
            content=self.responses.pop(0),
        )


def executor_llm_for(*, tool_name, tool_input, candidate_facts, summary="Tool actions returned structured evidence."):
    return SequencedExecutorLLM(
        [
            _tool_call(tool_name, tool_input),
            _final_result(tool_name, tool_input, candidate_facts, summary=summary),
        ]
    )


def _tool_call(tool_name, tool_input):
    return f"""{{
  "schema_version": "executor.v1",
  "agent_name": "executor_agent",
  "action_type": "tool_call",
  "node_id": "task_001",
  "tool_name": "{tool_name}",
  "tool_input": {tool_input},
  "rationale": "Collect evidence."
}}"""


def _final_result(tool_name, tool_input, candidate_facts, summary="Tool actions returned structured evidence."):
    return f"""{{
  "schema_version": "executor.v1",
  "agent_name": "executor_agent",
  "action_type": "final_result",
  "node_id": "task_001",
  "status": "success",
  "summary": "{summary}",
  "actions": [
    {{
      "tool_name": "{tool_name}",
      "tool_input": {tool_input},
      "output_ref": "artifact_001",
      "status": "success",
      "error": null
    }}
  ],
  "candidate_facts": {candidate_facts},
  "evidence_refs": ["artifact_001"],
  "errors": [],
  "confidence": 0.9
}}"""


def make_blackboard(target="http://127.0.0.1:8080"):
    return SharedBlackboard(
        context=TaskContext(
            task_id="pentest_001",
            goal="Validate executor flow",
            target=target,
            scope=[target],
        )
    )


def test_execution_router_selects_registered_agent_by_task_type():
    decision = ExecutionRouter().route(
        TaskNode(
            node_id="task_001",
            task_type="web_fingerprint",
            description="Identify web stack",
        )
    )

    assert decision.selected_agent == "web_agent"
    assert decision.blocked is False
    assert "http_probe" in decision.required_capabilities


def test_execution_router_blocks_explicit_executor_capability_mismatch():
    decision = ExecutionRouter().route(
        TaskNode(
            node_id="task_001",
            task_type="host_info",
            description="Collect host info",
            assigned_executor="web_agent",
        )
    )

    assert decision.blocked is True
    assert decision.block_reason == "assigned_executor_capability_mismatch"
    assert decision.selected_agent == "planner_agent"


def test_recon_agent_records_actions_service_fact_and_artifact():
    blackboard = make_blackboard()
    node = TaskNode(
        node_id="task_001",
        task_type="service_detection",
        description="Detect service",
        target="http://127.0.0.1:8080",
    )
    blackboard.add_node(node)

    llm = executor_llm_for(
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
    result = ReconAgent(FakeTools(), llm_client=llm).execute(node, blackboard)

    assert result.status == "success"
    assert result.summary == "Tool actions returned structured evidence."
    assert result.actions[0]["tool_name"] == "http_probe"
    assert result.candidate_facts[0]["type"] == "service"
    assert result.candidate_facts[0]["value"] == "http"
    assert result.candidate_facts[0]["reason"]
    assert result.evidence_refs[0] in blackboard.artifacts
    assert blackboard.execution_results[-1] is result
    assert blackboard.graph.nodes["task_001"].status == "success"


def test_llm_executor_runs_single_tool_call_then_final_result():
    blackboard = make_blackboard()
    node = TaskNode(
        node_id="task_001",
        task_type="service_detection",
        description="Detect service",
        target="http://127.0.0.1:8080",
        success_criteria=["Collect service evidence."],
    )
    blackboard.add_node(node)

    class FakeExecutorLLM:
        def __init__(self):
            self.calls = 0

        def generate(self, request):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    provider="fake",
                    model="fake-executor",
                    content="""{
  "schema_version": "executor.v1",
  "agent_name": "recon_agent",
  "action_type": "tool_call",
  "node_id": "task_001",
  "tool_name": "http_probe",
  "tool_input": {"url": "http://127.0.0.1:8080"},
  "rationale": "Collect response evidence."
}""",
                )
            return LLMResponse(
                provider="fake",
                model="fake-executor",
                content="""{
  "schema_version": "executor.v1",
  "agent_name": "recon_agent",
  "action_type": "final_result",
  "node_id": "task_001",
  "status": "success",
  "summary": "HTTP response evidence was collected.",
  "actions": [
    {
      "tool_name": "http_probe",
      "tool_input": {"url": "http://127.0.0.1:8080"},
      "output_ref": "artifact_001",
      "status": "success",
      "error": null
    }
  ],
  "candidate_facts": [
    {
      "type": "service",
      "target": "http://127.0.0.1:8080",
      "key": "protocol",
      "value": "http",
      "confidence": 0.9,
      "evidence_refs": ["artifact_001"],
      "reason": "HTTP probe returned a response."
    }
  ],
  "evidence_refs": ["artifact_001"],
  "errors": [],
  "confidence": 0.9
}""",
            )

    llm = FakeExecutorLLM()
    result = ReconAgent(FakeTools(), llm_client=llm).execute(node, blackboard)

    assert llm.calls == 2
    assert result.status == "success"
    assert result.actions[0]["tool_name"] == "http_probe"
    assert result.evidence_refs == ["artifact_001"]
    assert result.candidate_facts[0]["reason"]
    assert blackboard.graph.nodes["task_001"].status == "success"


def test_web_agent_extracts_technology_candidate_facts():
    blackboard = make_blackboard()
    node = TaskNode(
        node_id="task_002",
        task_type="web_fingerprint",
        description="Identify web stack",
        target="http://127.0.0.1:8080",
    )
    blackboard.add_node(node)

    llm = executor_llm_for(
        tool_name="http_probe",
        tool_input='{"url": "http://127.0.0.1:8080"}',
        candidate_facts="""[
    {"type": "technology", "target": "http://127.0.0.1:8080", "key": "server", "value": "nginx", "confidence": 0.7, "evidence_refs": ["artifact_001"], "reason": "server header"},
    {"type": "technology", "target": "http://127.0.0.1:8080", "key": "x_powered_by", "value": "Python", "confidence": 0.7, "evidence_refs": ["artifact_001"], "reason": "x-powered-by header"},
    {"type": "technology", "target": "http://127.0.0.1:8080", "key": "title", "value": "Local Lab", "confidence": 0.7, "evidence_refs": ["artifact_001"], "reason": "page title"}
  ]""",
    )
    result = WebAgent(FakeTools(), llm_client=llm).execute(node, blackboard)

    assert result.status == "success"
    values = {fact["value"] for fact in result.candidate_facts}
    assert {"nginx", "Python", "Local Lab"} <= values


def test_host_agent_runs_host_info_for_configured_target():
    blackboard = make_blackboard(target="http://example.com")
    node = TaskNode(
        node_id="task_003",
        task_type="host_info",
        description="Collect host info",
        target="http://example.com",
    )
    blackboard.add_node(node)

    llm = executor_llm_for(
        tool_name="host_info",
        tool_input="{}",
        candidate_facts="""[
    {"type": "host", "target": "http://example.com", "key": "platform", "value": "Linux-test", "confidence": 0.9, "evidence_refs": ["artifact_001"], "reason": "host_info returned platform"}
  ]""",
    )
    result = HostAgent(FakeTools(), llm_client=llm).execute(node, blackboard)

    assert result.status == "success"
    assert result.errors == []
    assert result.candidate_facts[0]["type"] == "host"
    assert blackboard.artifacts[result.evidence_refs[0]].content["status"] == "success"
