from dataclasses import dataclass, field
from typing import Any, Mapping

from divine.models.task import TaskNode


@dataclass(frozen=True)
class AgentCapability:
    agent_name: str
    role: str
    supported_task_types: list[str]
    tools: list[str]
    description: str


@dataclass(frozen=True)
class RouteDecision:
    selected_agent: str
    routing_reason: str
    required_capabilities: list[str] = field(default_factory=list)
    risk_level: str = "low"
    blocked: bool = False
    block_reason: str | None = None


class CapabilityRegistry:
    def __init__(self, capabilities: list[AgentCapability] | None = None) -> None:
        self.capabilities = {
            item.agent_name: item
            for item in (capabilities or default_capabilities())
        }

    def agent_for_task_type(self, task_type: str) -> AgentCapability | None:
        for capability in self.capabilities.values():
            if task_type in capability.supported_task_types:
                return capability
        for capability in self.capabilities.values():
            if task_type.endswith("_validation") and task_type.removesuffix("_validation") in capability.supported_task_types:
                return capability
        return None


class ExecutionRouter:
    def __init__(self, registry: CapabilityRegistry | None = None) -> None:
        self.registry = registry or CapabilityRegistry()

    def route(self, task: TaskNode | Mapping[str, Any]) -> RouteDecision:
        task_type = _task_type(task)
        assigned_executor = _assigned_executor(task)
        risk_level = _risk_level(task)
        if assigned_executor:
            capability = self.registry.capabilities.get(assigned_executor)
            if capability is None:
                return RouteDecision(
                    selected_agent="planner_agent",
                    routing_reason=f"Assigned executor is not registered: {assigned_executor}.",
                    risk_level=risk_level,
                    blocked=True,
                    block_reason="unknown_assigned_executor",
                )
            if not _capability_supports_task(capability, task_type):
                return RouteDecision(
                    selected_agent="planner_agent",
                    routing_reason=f"Assigned executor {assigned_executor} does not support task_type={task_type}.",
                    risk_level=risk_level,
                    blocked=True,
                    block_reason="assigned_executor_capability_mismatch",
                )
            return RouteDecision(
                selected_agent=capability.agent_name,
                routing_reason="Task has an explicit executor assignment.",
                required_capabilities=capability.tools,
                risk_level=risk_level,
            )

        capability = self.registry.agent_for_task_type(task_type)
        if capability is None:
            return RouteDecision(
                selected_agent="planner_agent",
                routing_reason=f"No executor capability registered for task_type={task_type}.",
                risk_level=risk_level,
                blocked=True,
                block_reason="unsupported_task_type",
            )
        return RouteDecision(
            selected_agent=capability.agent_name,
            routing_reason=f"Matched task_type={task_type}.",
            required_capabilities=capability.tools,
            risk_level=risk_level,
        )


def default_capabilities() -> list[AgentCapability]:
    return [
        AgentCapability(
            agent_name="recon_agent",
            role="Reconnaissance executor",
            supported_task_types=["recon", "service_detection", "port_scan", "http_probe"],
            tools=["tcp_connect_check", "http_probe", "https_probe"],
            description="Discovers reachable services and collects low-risk target facts.",
        ),
        AgentCapability(
            agent_name="web_agent",
            role="Web testing executor",
            supported_task_types=["web", "web_fingerprint", "web_path_discovery", "web_validation"],
            tools=["http_probe", "https_probe", "path_probe"],
            description="Inspects HTTP applications, paths, response metadata, and web evidence.",
        ),
        AgentCapability(
            agent_name="host_agent",
            role="Host inspection executor",
            supported_task_types=["host", "host_info"],
            tools=["host_info", "run_command"],
            description="Collects local host and environment information for authorized targets.",
        ),
    ]


def _capability_supports_task(capability: AgentCapability, task_type: str) -> bool:
    return (
        task_type in capability.supported_task_types
        or task_type.endswith("_validation")
        and task_type.removesuffix("_validation") in capability.supported_task_types
    )


def _task_type(task: TaskNode | Mapping[str, Any]) -> str:
    if isinstance(task, Mapping):
        return str(task.get("task_type") or task.get("executor_type") or "unknown")
    if task.result and isinstance(task.result, Mapping) and task.result.get("task_type"):
        return str(task.result["task_type"])
    if task.executor_type:
        return task.executor_type.value
    return "unknown"


def _assigned_executor(task: TaskNode | Mapping[str, Any]) -> str | None:
    if isinstance(task, Mapping):
        value = task.get("assigned_executor")
        return str(value) if value else None
    if task.result and isinstance(task.result, Mapping):
        value = task.result.get("assigned_executor")
        return str(value) if value else None
    return None


def _risk_level(task: TaskNode | Mapping[str, Any]) -> str:
    if isinstance(task, Mapping):
        return str(task.get("risk_level") or "low")
    if task.result and isinstance(task.result, Mapping):
        return str(task.result.get("risk_level") or "low")
    return "low"

