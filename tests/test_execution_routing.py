from divine.agents.routing import ExecutionRouter
from divine.models.common import ExecutorType, PentestPhase
from divine.models.task import TaskNode


def test_execution_router_selects_agent_from_task_executor_type():
    task = TaskNode(
        id="web_1",
        description="Fingerprint web stack",
        phase=PentestPhase.RECON,
        executor_type=ExecutorType.WEB,
    )

    decision = ExecutionRouter().route(task)

    assert decision.selected_agent == "web_agent"
    assert decision.blocked is False
    assert "http_probe" in decision.required_capabilities


def test_execution_router_selects_agent_from_mapping_task_type():
    decision = ExecutionRouter().route(
        {
            "id": "service_1",
            "task_type": "service_detection",
            "risk_level": "medium",
        }
    )

    assert decision.selected_agent == "recon_agent"
    assert decision.risk_level == "medium"
    assert decision.blocked is False


def test_execution_router_blocks_explicit_capability_mismatch():
    decision = ExecutionRouter().route(
        {
            "id": "host_1",
            "task_type": "host_info",
            "assigned_executor": "web_agent",
        }
    )

    assert decision.blocked is True
    assert decision.block_reason == "assigned_executor_capability_mismatch"
    assert decision.selected_agent == "planner_agent"


def test_execution_router_blocks_unknown_task_type():
    decision = ExecutionRouter().route({"id": "x", "task_type": "unsupported"})

    assert decision.blocked is True
    assert decision.block_reason == "unsupported_task_type"

