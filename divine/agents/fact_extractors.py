from typing import Any, Mapping
from urllib.parse import urlparse

from divine.tools import ToolResult


def recon_candidate_facts(
    *,
    task: Mapping[str, Any],
    tool_results: list[ToolResult],
    execution_id: str,
    evidence_refs: list[str],
) -> list[Mapping[str, Any]]:
    facts: list[Mapping[str, Any]] = []
    for index, result in enumerate(tool_results):
        evidence = [evidence_refs[index]] if index < len(evidence_refs) else []
        if result.tool_name in {"http_probe", "https_probe"} and result.succeeded:
            url = str(result.output.get("url") or result.input.get("url") or task.get("target") or "")
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


def web_candidate_facts(
    *,
    task: Mapping[str, Any],
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
                        "target": str(result.output.get("url") or result.input.get("url") or task.get("target") or ""),
                        "key": key,
                        "value": value,
                        "confidence": 0.7,
                        "source": execution_id,
                        "evidence_refs": evidence,
                        "reason": f"{key} was present in the web response evidence.",
                    }
                )
        if result.tool_name == "path_probe":
            discovered = result.output.get("discovered") if isinstance(result.output.get("discovered"), list) else []
            for item in discovered:
                if not isinstance(item, Mapping):
                    continue
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


def host_candidate_facts(
    *,
    task: Mapping[str, Any],
    tool_results: list[ToolResult],
    execution_id: str,
    evidence_refs: list[str],
) -> list[Mapping[str, Any]]:
    if not tool_results or not tool_results[0].succeeded:
        return []
    return [
        {
            "type": "host",
            "target": task.get("target") or "local",
            "key": "platform",
            "value": tool_results[0].output.get("platform"),
            "confidence": 0.9,
            "source": execution_id,
            "evidence_refs": evidence_refs[:1],
            "reason": "host_info returned platform information.",
        }
    ]


def target_parts(target: str) -> tuple[str, int, str | None, str]:
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


def ensure_url(target: str) -> str:
    if "://" in target:
        return target
    host, port, _, _ = target_parts(target)
    return f"http://{host}:{port}"

