from divine.agents.fact_extractors import (
    ensure_url,
    host_candidate_facts,
    recon_candidate_facts,
    target_parts,
    web_candidate_facts,
)
from divine.tools import ToolResult


def test_recon_candidate_facts_extracts_http_protocol():
    result = ToolResult(
        "http_probe",
        "success",
        {"url": "http://127.0.0.1:8080"},
        {"url": "http://127.0.0.1:8080"},
        artifact_type="http_response",
    )

    facts = recon_candidate_facts(
        task={"target": "http://127.0.0.1:8080"},
        tool_results=[result],
        execution_id="exec_001",
        evidence_refs=["artifact_001"],
    )

    assert facts[0]["type"] == "service"
    assert facts[0]["key"] == "protocol"
    assert facts[0]["value"] == "http"
    assert facts[0]["evidence_refs"] == ["artifact_001"]


def test_web_candidate_facts_extracts_headers_title_and_paths():
    response = ToolResult(
        "http_probe",
        "success",
        {"url": "http://127.0.0.1:8080"},
        {
            "url": "http://127.0.0.1:8080",
            "headers": {"server": "nginx", "x-powered-by": "Python"},
            "title": "Local Lab",
        },
        artifact_type="http_response",
    )
    paths = ToolResult(
        "path_probe",
        "success",
        {"base_url": "http://127.0.0.1:8080"},
        {"discovered": [{"path": "/admin", "url": "http://127.0.0.1:8080/admin"}]},
        artifact_type="web_path_probe",
    )

    facts = web_candidate_facts(
        task={"target": "http://127.0.0.1:8080"},
        tool_results=[response, paths],
        execution_id="exec_001",
        evidence_refs=["artifact_001", "artifact_002"],
    )

    values = {fact["value"] for fact in facts}
    assert {"nginx", "Python", "Local Lab", "/admin"} <= values
    assert all(fact["evidence_refs"] for fact in facts)


def test_host_candidate_facts_extracts_platform():
    result = ToolResult("host_info", "success", {}, {"platform": "Linux-test"}, artifact_type="host_info")

    facts = host_candidate_facts(
        task={"target": "local"},
        tool_results=[result],
        execution_id="exec_001",
        evidence_refs=["artifact_001"],
    )

    assert facts == [
        {
            "type": "host",
            "target": "local",
            "key": "platform",
            "value": "Linux-test",
            "confidence": 0.9,
            "source": "exec_001",
            "evidence_refs": ["artifact_001"],
            "reason": "host_info returned platform information.",
        }
    ]


def test_target_parts_and_ensure_url_defaults():
    assert target_parts("example.com:8080") == ("example.com", 8080, None, "http://example.com:8080")
    assert target_parts("https://example.com") == ("example.com", 443, "https", "https://example.com")
    assert ensure_url("example.com") == "http://example.com:80"

