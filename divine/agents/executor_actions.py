import json
import time
from typing import Any, Mapping

from divine.tools import ToolAdapter, ToolResult


def parse_executor_action(content: str) -> Mapping[str, Any]:
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


def run_tool_action(action: Mapping[str, Any], tools: ToolAdapter | None = None) -> ToolResult:
    adapter = tools or ToolAdapter()
    tool_name = str(action.get("tool_name") or "")
    tool_input = _mapping(action.get("tool_input"))
    if tool_name not in tool_catalog():
        return ToolResult(tool_name or "unknown_tool", "failed", tool_input, {}, error="unknown_tool")
    try:
        if tool_name == "tcp_connect_check":
            return adapter.tcp_connect_check(
                str(tool_input.get("host") or ""),
                int(tool_input.get("port") or 0),
                timeout=float(tool_input.get("timeout") or 2.0),
            )
        if tool_name == "http_probe":
            return adapter.http_probe(
                str(tool_input.get("url") or ""),
                timeout=float(tool_input.get("timeout") or 5.0),
            )
        if tool_name == "https_probe":
            return adapter.https_probe(
                str(tool_input.get("url") or ""),
                timeout=float(tool_input.get("timeout") or 5.0),
            )
        if tool_name == "path_probe":
            paths = tool_input.get("paths")
            return adapter.path_probe(
                str(tool_input.get("base_url") or ""),
                paths=list(paths) if isinstance(paths, list) else None,
                timeout=float(tool_input.get("timeout") or 5.0),
            )
        if tool_name == "host_info":
            return adapter.host_info()
    except Exception as exc:
        return ToolResult(tool_name, "failed", tool_input, {}, error=str(exc))
    return ToolResult(tool_name, "failed", tool_input, {}, error="unsupported_tool")


def run_timed_tool_action(
    action: Mapping[str, Any],
    tools: ToolAdapter | None = None,
) -> tuple[ToolResult, float]:
    started = time.perf_counter()
    result = run_tool_action(action, tools)
    return result, round((time.perf_counter() - started) * 1000, 3)


def observation_from_tool_result(
    result: ToolResult,
    *,
    evidence_ref: str | None = None,
    duration_ms: float | None = None,
) -> dict[str, Any]:
    return {
        "tool_name": result.tool_name,
        "tool_input": dict(result.input),
        "status": result.status,
        "output": dict(result.output),
        "error": result.error,
        "evidence_ref": evidence_ref,
        "duration_ms": duration_ms,
    }


def tool_catalog() -> set[str]:
    return {"tcp_connect_check", "http_probe", "https_probe", "path_probe", "host_info"}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}

