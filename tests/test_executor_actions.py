from divine.agents.executor_actions import (
    observation_from_tool_result,
    parse_executor_action,
    run_tool_action,
    tool_catalog,
)
from divine.tools import ToolResult


class FakeTools:
    def http_probe(self, url, *, timeout=5.0):
        return ToolResult(
            "http_probe",
            "success",
            {"url": url, "timeout": timeout},
            {"status_code": 200, "title": "Local Lab"},
            artifact_type="http_response",
        )


def test_parse_executor_action_extracts_json_from_markdown():
    action = parse_executor_action(
        """tool call:
```json
{"action_type": "tool_call", "tool_name": "http_probe", "tool_input": {"url": "http://127.0.0.1"}}
```
"""
    )

    assert action["action_type"] == "tool_call"
    assert action["tool_name"] == "http_probe"


def test_run_tool_action_dispatches_registered_tool():
    result = run_tool_action(
        {
            "tool_name": "http_probe",
            "tool_input": {"url": "http://127.0.0.1", "timeout": 1},
        },
        FakeTools(),
    )

    assert result.succeeded is True
    assert result.output["title"] == "Local Lab"
    assert result.artifact_type == "http_response"


def test_run_tool_action_rejects_unknown_tool():
    result = run_tool_action({"tool_name": "dangerous_tool", "tool_input": {"cmd": "id"}})

    assert result.succeeded is False
    assert result.error == "unknown_tool"


def test_observation_from_tool_result_shapes_executor_context():
    result = ToolResult("host_info", "success", {}, {"system": "Linux"}, artifact_type="host_info")

    observation = observation_from_tool_result(
        result,
        evidence_ref="artifact_001",
        duration_ms=12.5,
    )

    assert observation["tool_name"] == "host_info"
    assert observation["evidence_ref"] == "artifact_001"
    assert observation["duration_ms"] == 12.5
    assert "host_info" in tool_catalog()

