from divine.models.audit import (
    AuditFeedback,
    AuditResult,
    FailureAttribution,
    PlanningFeedback,
    TaskJudgement,
)


def test_audit_feedback_defaults_capture_evaluator_contract():
    feedback = AuditFeedback(
        feedback_id="fb_001",
        task_id="task_001",
        task_judgement=TaskJudgement(status="success", completion_score=1.0, confidence=0.9),
        audit_result=AuditResult(
            confirmed_facts=[{"type": "service", "key": "protocol", "value": "http"}],
            evidence_refs=["artifact_001"],
        ),
        failure_attribution=FailureAttribution(level="none"),
        planning_feedback=PlanningFeedback(recommended_strategy="expand"),
    )

    assert feedback.task_judgement.status == "success"
    assert feedback.audit_result.confirmed_facts[0]["value"] == "http"
    assert feedback.failure_attribution.level == "none"
    assert feedback.planning_feedback.recommended_strategy == "expand"

