from pathlib import Path

from divine.context import ContextSection
from divine.logger.config import LLMTraceSettings
from divine.prompts import PromptRenderer
from divine.prompts.renderer import DEFAULT_TEMPLATE_DIR


TRACE_SETTINGS = LLMTraceSettings(enabled=False)


def renderer() -> PromptRenderer:
    return PromptRenderer(trace_settings=TRACE_SETTINGS)


def test_shared_global_static_renders_and_includes_core_sections():
    rendered = renderer().render(template_id="shared.global_static", variables={})

    assert "Output discipline" in rendered.content
    assert "shared/output_discipline.j2" in rendered.referenced_templates
    assert "shared/safety_boundaries.j2" not in rendered.referenced_templates
    assert "shared/risk_model.j2" not in rendered.referenced_templates


def test_mission_scope_template_requires_objective_and_renders_notes():
    rendered = renderer().render(
        template_id="missions.scope",
        variables={
            "target": "http://127.0.0.1:8080",
            "scope": "unused in prompt",
            "objective": "Validate the framework loop",
            "constraints": ["Keep outputs concise", "Prefer structured context"],
        },
    )

    assert "Target" in rendered.content
    assert "Validate the framework loop" in rendered.content
    assert "Keep outputs concise" in rendered.content
    assert "unused in prompt" not in rendered.content


def test_agent_static_templates_render_without_dynamic_variables():
    template_ids = [
        "agents.planner.role",
        "agents.planner.output_contract",
        "agents.router.role",
        "agents.router.output_contract",
        "agents.executor.base_role",
        "agents.executor.recon_role",
        "agents.executor.web_role",
        "agents.executor.host_role",
        "agents.executor.output_contract",
        "agents.evaluator.role",
        "agents.evaluator.output_contract",
    ]

    for template_id in template_ids:
        rendered = renderer().render(template_id=template_id, variables={})
        assert rendered.content.strip(), template_id


def test_agent_task_templates_render_with_sample_variables():
    sample_evidence = [{"id": "ev-1", "summary": "Local service returned 200"}]
    samples = {
        "agents.planner.initial_dag": {
            "target": "http://127.0.0.1:8080",
            "scope": "unused in prompt",
            "objective": "Generate the initial DAG",
            "blackboard_summary": "No nodes yet",
        },
        "agents.planner.evolve_dag": {
            "dag_state": "node-1 failed",
            "execution_summary": "Missing input",
            "audit_feedback": "missing_input",
        },
        "agents.router.route_task": {
            "node": {"node_id": "node-1", "objective": "Identify entry points"},
        },
        "agents.executor.execute_node": {
            "node": {"node_id": "node-1", "objective": "Identify entry points"},
            "inputs": {"target": "http://127.0.0.1:8080"},
            "success_criteria": ["List entry points"],
        },
        "agents.evaluator.audit": {
            "target": "http://127.0.0.1:8080",
            "node": {"node_id": "node-1"},
            "success_criteria": ["List entry points"],
            "execution_result": {"status": "partial"},
            "evidence": sample_evidence,
            "blackboard_summary": "No confirmed facts yet",
        },
        "runtime.environment": {
            "current_time": "2026-04-27T00:00:00Z",
            "cwd": "/tmp/project",
            "is_git_repo": True,
            "os_name": "Linux",
            "shell": "bash",
            "provider": "openai",
            "model": "gpt-5.5",
        },
        "runtime.blackboard_summary": {
            "blackboard_summary": "Completed 1 node",
            "dag_state": "node-1 done",
            "evidence": sample_evidence,
            "audit_feedback": [{"id": "audit-1", "summary": "Partially passed"}],
        },
    }

    for template_id, variables in samples.items():
        rendered = renderer().render(template_id=template_id, variables=variables)
        assert rendered.content.strip(), template_id


def test_render_agent_role_as_cacheable_prompt_segment():
    segment = renderer().render_segment(
        template_id="agents.planner.role",
        section=ContextSection.STATIC,
        stable=True,
        cache_policy="explicit",
        agent="planner",
    )

    assert segment.name == "agents.planner.role"
    assert segment.section == ContextSection.STATIC
    assert segment.stable is True
    assert "Planner Agent" in segment.content


def test_all_jinja_templates_are_covered_by_render_tests():
    templates_dir = DEFAULT_TEMPLATE_DIR
    template_paths = {
        path.relative_to(templates_dir).as_posix()
        for path in templates_dir.rglob("*.j2")
    }
    intentionally_included_only = {
        "shared/safety_boundaries.j2",
        "shared/output_discipline.j2",
        "shared/risk_model.j2",
        "shared/json_response_rules.j2",
    }
    rendered_or_included = {
        "shared/global_static.j2",
        "missions/scope.j2",
        "agents/planner/role.j2",
        "agents/planner/output_contract.j2",
        "agents/planner/initial_dag.j2",
        "agents/planner/evolve_dag.j2",
        "agents/router/role.j2",
        "agents/router/output_contract.j2",
        "agents/router/route_task.j2",
        "agents/executor/base_role.j2",
        "agents/executor/recon_role.j2",
        "agents/executor/web_role.j2",
        "agents/executor/host_role.j2",
        "agents/executor/output_contract.j2",
        "agents/executor/execute_node.j2",
        "agents/evaluator/role.j2",
        "agents/evaluator/output_contract.j2",
        "agents/evaluator/audit.j2",
        "runtime/environment.j2",
        "runtime/blackboard_summary.j2",
    }

    assert template_paths == rendered_or_included | intentionally_included_only
