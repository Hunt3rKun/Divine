from divine.models.execution import execution_result_from_final_action


def test_execution_result_from_final_action_keeps_valid_refs_and_errors():
    action = {
        "status": "success",
        "summary": "HTTP response evidence was collected.",
        "actions": [{"tool_name": "http_probe", "output_ref": "artifact_001"}],
        "candidate_facts": [{"type": "service", "key": "protocol", "value": "http"}],
        "evidence_refs": ["artifact_001", "missing"],
        "errors": ["tool_warning"],
        "confidence": 0.9,
    }

    result = execution_result_from_final_action(
        action,
        execution_id="exec_001",
        task_id="task_001",
        executor="recon_agent",
        fallback_evidence_refs=["artifact_001"],
        fallback_errors=["previous_warning"],
    )

    assert result.status == "success"
    assert result.summary.startswith("HTTP")
    assert result.evidence_refs == ["artifact_001"]
    assert result.raw_output_refs == ["artifact_001"]
    assert result.errors == ["previous_warning", "tool_warning"]
    assert result.candidate_facts[0]["value"] == "http"
    assert result.confidence == 0.9


def test_execution_result_from_final_action_falls_back_to_observed_refs():
    action = {
        "status": "unknown",
        "evidence_refs": ["not_allowed"],
        "confidence": True,
    }

    result = execution_result_from_final_action(
        action,
        execution_id="exec_001",
        task_id="task_001",
        executor="web_agent",
        fallback_evidence_refs=["artifact_001", "artifact_002"],
    )

    assert result.status == "failed"
    assert result.evidence_refs == ["artifact_001", "artifact_002"]
    assert result.confidence == 0.0

