"""Executor agents and routing for node-level task execution."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol
from urllib.parse import urlparse

from divine.blackboard import Artifact, ExecutionResult, SharedBlackboard, TaskNode
from divine.llm.types import LLMRequest, Message
from divine.logger import get_logger
from divine.prompts import PromptRenderer
from divine.tools import ToolAdapter, ToolResult


class LLMGenerator(Protocol):
    def generate(self, request: LLMRequest) -> Any:
        ...


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
        self.capabilities = {item.agent_name: item for item in (capabilities or default_capabilities())}

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

    def route(self, node: TaskNode) -> RouteDecision:
        if node.assigned_executor:
            capability = self.registry.capabilities.get(node.assigned_executor)
            if capability is None:
                return RouteDecision(
                    selected_agent="planner_agent",
                    routing_reason=f"Assigned executor is not registered: {node.assigned_executor}.",
                    risk_level=node.risk_level,
                    blocked=True,
                    block_reason="unknown_assigned_executor",
                )
            if not _capability_supports_task(capability, node.task_type):
                return RouteDecision(
                    selected_agent="planner_agent",
                    routing_reason=(
                        f"Assigned executor {node.assigned_executor} does not support task_type={node.task_type}."
                    ),
                    risk_level=node.risk_level,
                    blocked=True,
                    block_reason="assigned_executor_capability_mismatch",
                )
            return RouteDecision(
                selected_agent=capability.agent_name,
                routing_reason="Node has an explicit executor assignment.",
                required_capabilities=capability.tools,
                risk_level=node.risk_level,
            )
        capability = self.registry.agent_for_task_type(node.task_type)
        if not capability:
            return RouteDecision(
                selected_agent="planner_agent",
                routing_reason=f"No executor capability registered for task_type={node.task_type}.",
                risk_level=node.risk_level,
                blocked=True,
                block_reason="unsupported_task_type",
            )
        return RouteDecision(
            selected_agent=capability.agent_name,
            routing_reason=f"Matched task_type={node.task_type}.",
            required_capabilities=capability.tools,
            risk_level=node.risk_level,
        )


class ExecutorAgent:
    agent_name = "executor_agent"
    supported_task_types: set[str] = set()

    def __init__(
        self,
        tools: ToolAdapter | None = None,
        *,
        llm_client: LLMGenerator,
        renderer: PromptRenderer | None = None,
        max_turns: int = 5,
        max_tokens: int = 4096,
    ) -> None:
        self.tools = tools or ToolAdapter()
        self.llm_client = llm_client
        self.renderer = renderer or PromptRenderer()
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self._log = get_logger("executor")

    def execute(self, node: TaskNode, blackboard: SharedBlackboard) -> ExecutionResult:
        return self._execute_llm_loop(node, blackboard)

    def _execute_tools(self, node: TaskNode, blackboard: SharedBlackboard) -> list[ToolResult]:
        raise NotImplementedError

    def _candidate_facts(
        self,
        node: TaskNode,
        tool_results: list[ToolResult],
        execution_id: str,
        evidence_refs: list[str],
    ) -> list[Mapping[str, Any]]:
        return []

    def _summarize_execution(
        self,
        node: TaskNode,
        tool_results: list[ToolResult],
        errors: list[str],
    ) -> tuple[str, float, str]:
        success = any(result.succeeded for result in tool_results)
        blocked = any(result.status == "blocked" for result in tool_results)
        if blocked:
            return "failed", 0.2, "Execution stopped because a tool returned blocked."
        if success:
            return "success", 0.8, "Tool actions returned structured evidence."
        return "failed", 0.2, "No tool action returned usable evidence."

    def _store_tool_result(
        self,
        result: ToolResult,
        node: TaskNode,
        blackboard: SharedBlackboard,
        *,
        duration_ms: float | None = None,
    ) -> str:
        artifact_id = blackboard.next_id("artifact")
        blackboard.add_artifact(
            Artifact(
                artifact_id=artifact_id,
                artifact_type=result.artifact_type,
                source=result.tool_name,
                node_id=node.node_id,
                content={
                    "tool_name": result.tool_name,
                    "status": result.status,
                    "input": dict(result.input),
                    "output": dict(result.output),
                    "error": result.error,
                    "duration_ms": duration_ms,
                },
            )
        )
        return artifact_id

    def _execute_llm_loop(self, node: TaskNode, blackboard: SharedBlackboard) -> ExecutionResult:
        blackboard.record_event(
            "ExecutionStarted",
            f"Execution started for {node.node_id}",
            {"node_id": node.node_id, "executor": self.agent_name},
        )
        node.status = "running"
        execution_id = blackboard.next_id("exec")
        observations: list[Mapping[str, Any]] = []
        tool_results: list[ToolResult] = []
        evidence_refs: list[str] = []
        errors: list[str] = []

        self._log.info(
            "Executor started task_id={} node_id={} execution_id={} executor={} task_type={} max_turns={}",
            blackboard.context.task_id,
            node.node_id,
            execution_id,
            self.agent_name,
            node.task_type,
            self.max_turns,
        )

        for turn in range(1, self.max_turns + 1):
            request = self._build_action_request(node, blackboard, observations, execution_id=execution_id, turn=turn)
            response = self.llm_client.generate(request)  # type: ignore[union-attr]
            action = _parse_json_object(str(response.content))
            action_type = str(action.get("action_type") or "")
            self._log.debug(
                "Executor LLM action task_id={} node_id={} execution_id={} turn={} action_type={} trace_id={}",
                blackboard.context.task_id,
                node.node_id,
                execution_id,
                turn,
                action_type,
                request.trace_id,
            )
            if action_type == "tool_call":
                tool_name = str(action.get("tool_name") or "")
                self._log.info(
                    "Executor tool_call started task_id={} node_id={} execution_id={} turn={} tool_name={} tool_input={}",
                    blackboard.context.task_id,
                    node.node_id,
                    execution_id,
                    turn,
                    tool_name,
                    dict(_mapping(action.get("tool_input"))),
                )
                started = time.perf_counter()
                result = self._run_tool_call(action)
                duration_ms = round((time.perf_counter() - started) * 1000, 3)
                artifact_id = self._store_tool_result(result, node, blackboard, duration_ms=duration_ms)
                tool_results.append(result)
                evidence_refs.append(artifact_id)
                if result.error:
                    errors.append(result.error)
                observations.append(
                    {
                        "tool_name": result.tool_name,
                        "tool_input": dict(result.input),
                        "status": result.status,
                        "output": dict(result.output),
                        "error": result.error,
                        "evidence_ref": artifact_id,
                        "duration_ms": duration_ms,
                    }
                )
                self._log.info(
                    "Executor tool_call finished task_id={} node_id={} execution_id={} turn={} tool_name={} status={} artifact_id={} duration_ms={} error={}",
                    blackboard.context.task_id,
                    node.node_id,
                    execution_id,
                    turn,
                    result.tool_name,
                    result.status,
                    artifact_id,
                    duration_ms,
                    result.error,
                )
                continue
            if action_type == "final_result":
                result = self._execution_result_from_final_action(
                    action,
                    execution_id=execution_id,
                    node=node,
                    fallback_evidence_refs=evidence_refs,
                    fallback_errors=errors,
                )
                blackboard.add_execution_result(result)
                self._record_operation_trace(execution_id, node, blackboard, result)
                self._log.info(
                    "Executor final_result task_id={} node_id={} execution_id={} status={} evidence_refs={} errors={} confidence={}",
                    blackboard.context.task_id,
                    node.node_id,
                    execution_id,
                    result.status,
                    result.evidence_refs,
                    result.errors,
                    result.confidence,
                )
                return result
            raise ValueError(f"Unsupported executor action_type: {action_type}")

        result = ExecutionResult(
            execution_id=execution_id,
            node_id=node.node_id,
            executor=self.agent_name,
            status="failed",
            summary="Executor reached the maximum action turns without final_result.",
            actions=_actions_from_tool_results(tool_results, evidence_refs),
            tool_results=_tool_result_payloads(tool_results, evidence_refs),
            evidence_refs=evidence_refs,
            raw_output_refs=evidence_refs,
            errors=[*errors, "max_turns_exceeded"],
            confidence=0.0,
        )
        blackboard.add_execution_result(result)
        self._record_operation_trace(execution_id, node, blackboard, result)
        self._log.warning(
            "Executor max turns exceeded task_id={} node_id={} execution_id={} max_turns={} evidence_refs={} errors={}",
            blackboard.context.task_id,
            node.node_id,
            execution_id,
            self.max_turns,
            evidence_refs,
            result.errors,
        )
        return result

    def _build_action_request(
        self,
        node: TaskNode,
        blackboard: SharedBlackboard,
        observations: list[Mapping[str, Any]],
        *,
        execution_id: str,
        turn: int,
    ) -> LLMRequest:
        rendered = self.renderer.render(
            template_id="agents.executor.execute_node",
            variables={
                "node": _json_dumps(_node_payload(node)),
                "inputs": _json_dumps(
                    {
                        "target": node.target or blackboard.context.target,
                        "observations": observations,
                        "blackboard_summary": _blackboard_summary(blackboard),
                    }
                ),
                "success_criteria": node.success_criteria,
            },
            agent=self.agent_name,
        )
        role = self._role_prompt()
        contract = self.renderer.render(template_id="agents.executor.output_contract", variables={}, agent=self.agent_name)
        return LLMRequest(
            messages=[Message("user", rendered.content)],
            system=f"{role}\n\n{contract.content}",
            temperature=0,
            max_tokens=self.max_tokens,
            trace_id=rendered.trace_id,
            agent=self.agent_name,
            prompt_trace=rendered.as_trace(),
            trace_metadata={
                "task_id": blackboard.context.task_id,
                "node_id": node.node_id,
                "execution_id": execution_id,
                "turn": turn,
                "phase": "executor_action",
                "system_templates": [
                    "agents.executor.base_role",
                    f"agents.executor.{self.agent_name.removesuffix('_agent')}_role",
                    "agents.executor.output_contract",
                ],
            },
        )

    def _role_prompt(self) -> str:
        base = self.renderer.render(template_id="agents.executor.base_role", variables={}, agent=self.agent_name)
        role_template = {
            "recon_agent": "agents.executor.recon_role",
            "web_agent": "agents.executor.web_role",
            "host_agent": "agents.executor.host_role",
        }.get(self.agent_name)
        if not role_template:
            return base.content
        role = self.renderer.render(template_id=role_template, variables={}, agent=self.agent_name)
        return f"{base.content}\n\n{role.content}"

    def _run_tool_call(self, action: Mapping[str, Any]) -> ToolResult:
        tool_name = str(action.get("tool_name") or "")
        tool_input = _mapping(action.get("tool_input"))
        if tool_name not in self._tool_catalog():
            return ToolResult(tool_name or "unknown_tool", "failed", tool_input, {}, error="unknown_tool")
        try:
            if tool_name == "tcp_connect_check":
                return self.tools.tcp_connect_check(
                    str(tool_input.get("host") or ""),
                    int(tool_input.get("port") or 0),
                    timeout=float(tool_input.get("timeout") or 2.0),
                )
            if tool_name == "http_probe":
                return self.tools.http_probe(str(tool_input.get("url") or ""), timeout=float(tool_input.get("timeout") or 5.0))
            if tool_name == "https_probe":
                return self.tools.https_probe(str(tool_input.get("url") or ""), timeout=float(tool_input.get("timeout") or 5.0))
            if tool_name == "path_probe":
                paths = tool_input.get("paths")
                return self.tools.path_probe(
                    str(tool_input.get("base_url") or ""),
                    paths=list(paths) if isinstance(paths, list) else None,
                    timeout=float(tool_input.get("timeout") or 5.0),
                )
            if tool_name == "host_info":
                return self.tools.host_info()
        except Exception as exc:
            return ToolResult(tool_name, "failed", tool_input, {}, error=str(exc))
        return ToolResult(tool_name, "failed", tool_input, {}, error="unsupported_tool")

    def _tool_catalog(self) -> set[str]:
        return set()

    def _execution_result_from_final_action(
        self,
        action: Mapping[str, Any],
        *,
        execution_id: str,
        node: TaskNode,
        fallback_evidence_refs: list[str],
        fallback_errors: list[str],
    ) -> ExecutionResult:
        evidence_refs = _valid_refs(_string_list(action.get("evidence_refs")), fallback_evidence_refs)
        errors = [*fallback_errors, *_string_list(action.get("errors"))]
        return ExecutionResult(
            execution_id=execution_id,
            node_id=node.node_id,
            executor=self.agent_name,
            status=_execution_status(action.get("status")),
            summary=str(action.get("summary") or ""),
            actions=_mapping_list(action.get("actions")),
            candidate_facts=_mapping_list(action.get("candidate_facts")),
            evidence_refs=evidence_refs,
            raw_output_refs=evidence_refs,
            errors=[error for error in errors if error],
            confidence=_float(action.get("confidence")),
        )

    def _record_operation_trace(
        self,
        execution_id: str,
        node: TaskNode,
        blackboard: SharedBlackboard,
        result: ExecutionResult,
    ) -> None:
        blackboard.operation_traces.append(
            {
                "trace_id": execution_id,
                "agent_name": self.agent_name,
                "node_id": node.node_id,
                "action": node.task_type,
                "input": {"target": node.target or blackboard.context.target},
                "output_ref": result.evidence_refs,
                "status": result.status,
                "error": "; ".join(result.errors) or None,
            }
        )


class ReconAgent(ExecutorAgent):
    agent_name = "recon_agent"
    supported_task_types = {"asset_discovery", "port_check", "service_detection"}

    def _tool_catalog(self) -> set[str]:
        return {"tcp_connect_check", "http_probe", "https_probe"}

    def _execute_tools(self, node: TaskNode, blackboard: SharedBlackboard) -> list[ToolResult]:
        target = node.target or blackboard.context.target
        if node.task_type == "asset_discovery":
            return [
                ToolResult(
                    "asset_discovery",
                    "success",
                    {"target": target},
                    {"target": target},
                    artifact_type="target_profile",
                )
            ]
        host, port, scheme, url = _target_parts(target)
        if node.task_type == "port_check":
            return [self.tools.tcp_connect_check(host, port)]
        if scheme in {"http", "https"}:
            probe = self.tools.https_probe(url) if scheme == "https" else self.tools.http_probe(url)
            if probe.succeeded:
                return [probe]
            return [probe, self.tools.tcp_connect_check(host, port)]
        return [self.tools.tcp_connect_check(host, port)]

    def _candidate_facts(
        self,
        node: TaskNode,
        tool_results: list[ToolResult],
        execution_id: str,
        evidence_refs: list[str],
    ) -> list[Mapping[str, Any]]:
        facts: list[Mapping[str, Any]] = []
        for index, result in enumerate(tool_results):
            evidence = [evidence_refs[index]] if index < len(evidence_refs) else []
            if result.tool_name in {"http_probe", "https_probe"} and result.succeeded:
                url = str(result.output.get("url") or result.input.get("url") or node.target or "")
                scheme = urlparse(url).scheme or ("https" if result.tool_name == "https_probe" else "http")
                facts.append(
                    {
                        "type": "service",
                        "target": url,
                        "key": "protocol",
                        "value": scheme,
                        "confidence": 0.9,
                        "source": execution_id,
                        "evidence_refs": evidence,
                        "reason": "HTTP probe returned a successful response.",
                    }
                )
            if result.tool_name == "tcp_connect_check" and result.succeeded:
                facts.append(
                    {
                        "type": "service",
                        "target": f"{result.output.get('host')}:{result.output.get('port')}",
                        "key": "tcp_port",
                        "value": "open",
                        "confidence": 0.8,
                        "source": execution_id,
                        "evidence_refs": evidence,
                        "reason": "TCP connection succeeded for the host and port.",
                    }
                )
        return facts


class WebAgent(ExecutorAgent):
    agent_name = "web_agent"
    supported_task_types = {
        "web_fingerprint",
        "web_path_discovery",
        "web_rule_check",
        "response_header_analysis",
        "page_title_extraction",
        "static_resource_analysis",
        "service_detection_validation",
        "web_fingerprint_validation",
    }

    def _tool_catalog(self) -> set[str]:
        return {"http_probe", "https_probe", "path_probe"}

    def _execute_tools(self, node: TaskNode, blackboard: SharedBlackboard) -> list[ToolResult]:
        target = _ensure_url(node.target or blackboard.context.target)
        if node.task_type == "web_path_discovery":
            return [self.tools.path_probe(target)]
        return [self.tools.http_probe(target)]

    def _candidate_facts(
        self,
        node: TaskNode,
        tool_results: list[ToolResult],
        execution_id: str,
        evidence_refs: list[str],
    ) -> list[Mapping[str, Any]]:
        facts: list[Mapping[str, Any]] = []
        for index, result in enumerate(tool_results):
            if not result.succeeded:
                continue
            evidence = [evidence_refs[index]] if index < len(evidence_refs) else []
            headers = result.output.get("headers") if isinstance(result.output.get("headers"), Mapping) else {}
            server = headers.get("server") if isinstance(headers, Mapping) else None
            powered_by = headers.get("x-powered-by") if isinstance(headers, Mapping) else None
            for key, value in (("server", server), ("x_powered_by", powered_by), ("title", result.output.get("title"))):
                if value:
                    facts.append(
                        {
                            "type": "technology",
                            "target": str(result.output.get("url") or result.input.get("url") or node.target or ""),
                            "key": key,
                            "value": value,
                            "confidence": 0.7,
                            "source": execution_id,
                            "evidence_refs": evidence,
                            "reason": f"{key} was present in the web response evidence.",
                        }
                    )
            if result.tool_name == "path_probe":
                for item in result.output.get("discovered", []):
                    facts.append(
                        {
                            "type": "web_path",
                            "target": item.get("url"),
                            "key": "path",
                            "value": item.get("path"),
                            "confidence": 0.75,
                            "source": execution_id,
                            "evidence_refs": evidence,
                            "reason": "Path probe returned a discovered path observation.",
                        }
                    )
        return facts


class HostAgent(ExecutorAgent):
    agent_name = "host_agent"
    supported_task_types = {"host_info"}

    def _tool_catalog(self) -> set[str]:
        return {"host_info"}

    def _execute_tools(self, node: TaskNode, blackboard: SharedBlackboard) -> list[ToolResult]:
        return [self.tools.host_info()]

    def _candidate_facts(
        self,
        node: TaskNode,
        tool_results: list[ToolResult],
        execution_id: str,
        evidence_refs: list[str],
    ) -> list[Mapping[str, Any]]:
        if not tool_results or not tool_results[0].succeeded:
            return []
        return [
            {
                "type": "host",
                "target": node.target or "local",
                "key": "platform",
                "value": tool_results[0].output.get("platform"),
                "confidence": 0.9,
                "source": execution_id,
                "evidence_refs": evidence_refs[:1],
                "reason": "host_info returned platform information.",
            }
        ]


def default_capabilities() -> list[AgentCapability]:
    return [
        AgentCapability(
            "recon_agent",
            "information_gathering",
            ["asset_discovery", "port_check", "service_detection"],
            ["tcp_connect_check", "http_probe", "https_probe"],
            "Responsible for asset probing, port connectivity, and basic service detection.",
        ),
        AgentCapability(
            "web_agent",
            "web_testing",
            [
                "web_fingerprint",
                "web_path_discovery",
                "web_rule_check",
                "response_header_analysis",
                "page_title_extraction",
                "static_resource_analysis",
                "service_detection_validation",
                "web_fingerprint_validation",
            ],
            ["http_probe", "https_probe", "path_probe"],
            "Responsible for Web response, path, and fingerprint observations.",
        ),
        AgentCapability(
            "host_agent",
            "host_information",
            ["host_info"],
            ["host_info"],
            "Responsible for host information collection.",
        ),
    ]


def _capability_supports_task(capability: AgentCapability, task_type: str) -> bool:
    return task_type in capability.supported_task_types or (
        task_type.endswith("_validation")
        and task_type.removesuffix("_validation") in capability.supported_task_types
    )


def create_executor_agents(
    tools: ToolAdapter | None = None,
    *,
    llm_client: LLMGenerator,
) -> dict[str, ExecutorAgent]:
    return {
        "recon_agent": ReconAgent(tools, llm_client=llm_client),
        "web_agent": WebAgent(tools, llm_client=llm_client),
        "host_agent": HostAgent(tools, llm_client=llm_client),
    }


def _target_parts(target: str) -> tuple[str, int, str | None, str]:
    parsed = urlparse(target if "://" in target else f"//{target}")
    scheme = parsed.scheme or None
    host = parsed.hostname or target.split(":")[0]
    if parsed.port:
        port = parsed.port
    elif scheme == "https":
        port = 443
    else:
        port = 80
    if scheme:
        url = target
    else:
        url = f"http://{host}:{port}"
    return host, port, scheme, url


def _ensure_url(target: str) -> str:
    if "://" in target:
        return target
    host, port, _, _ = _target_parts(target)
    return f"http://{host}:{port}"


def _parse_json_object(content: str) -> Mapping[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Executor LLM response did not contain a JSON object") from None
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, Mapping):
        raise ValueError("Executor LLM response must be a JSON object")
    return payload


def _node_payload(node: TaskNode) -> Mapping[str, Any]:
    return {
        "node_id": node.node_id,
        "task_type": node.task_type,
        "description": node.description,
        "status": node.status,
        "dependencies": node.dependencies,
        "edge_type": node.edge_type,
        "risk_level": node.risk_level,
        "success_criteria": node.success_criteria,
        "assigned_executor": node.assigned_executor,
        "target": node.target,
        "metadata": node.metadata,
    }


def _blackboard_summary(blackboard: SharedBlackboard) -> Mapping[str, Any]:
    return {
        "confirmed_facts": [
            {
                "type": fact.type,
                "target": fact.target,
                "key": fact.key,
                "value": fact.value,
            }
            for fact in blackboard.intelligence.confirmed_facts
        ],
        "recent_events": [
            {"event_type": event.event_type, "message": event.message}
            for event in blackboard.event_log[-5:]
        ],
    }


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(float(value), 1.0))
    return 0.0


def _execution_status(value: Any) -> str:
    status = str(value) if value else "failed"
    allowed = {"success", "partial", "failed", "blocked", "needs_more_information"}
    return status if status in allowed else "failed"


def _valid_refs(refs: list[str], allowed_refs: list[str]) -> list[str]:
    return [ref for ref in refs if ref in allowed_refs] or list(allowed_refs)


def _actions_from_tool_results(
    tool_results: list[ToolResult],
    evidence_refs: list[str],
) -> list[Mapping[str, Any]]:
    return [
        {
            "tool_name": result.tool_name,
            "tool_input": dict(result.input),
            "output_ref": evidence_refs[index] if index < len(evidence_refs) else None,
            "status": result.status,
            "error": result.error,
        }
        for index, result in enumerate(tool_results)
    ]


def _tool_result_payloads(
    tool_results: list[ToolResult],
    evidence_refs: list[str],
) -> list[Mapping[str, Any]]:
    return [
        {
            "tool_name": result.tool_name,
            "status": result.status,
            "artifact_ref": evidence_refs[index] if index < len(evidence_refs) else None,
            "error": result.error,
        }
        for index, result in enumerate(tool_results)
    ]
