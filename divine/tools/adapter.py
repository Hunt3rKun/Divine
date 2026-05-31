"""Local tool adapter implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from platform import node as platform_node
from platform import platform, release, system
from socket import create_connection
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


@dataclass
class ToolResult:
    tool_name: str
    status: str
    input: Mapping[str, Any]
    output: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    artifact_type: str = "tool_output"

    @property
    def succeeded(self) -> bool:
        return self.status == "success"


class ToolAdapter:
    """Runs bounded checks and returns structured output."""

    def tcp_connect_check(self, host: str, port: int, *, timeout: float = 2.0) -> ToolResult:
        payload = {"host": host, "port": port, "timeout": timeout}
        try:
            with create_connection((host, port), timeout=timeout):
                return ToolResult(
                    tool_name="tcp_connect_check",
                    status="success",
                    input=payload,
                    output={"reachable": True, "host": host, "port": port},
                    artifact_type="tcp_connect",
                )
        except OSError as exc:
            return ToolResult(
                tool_name="tcp_connect_check",
                status="failed",
                input=payload,
                output={"reachable": False, "host": host, "port": port},
                error=str(exc),
                artifact_type="tcp_connect",
            )

    def http_probe(self, url: str, *, timeout: float = 5.0) -> ToolResult:
        return self._http_request("http_probe", url, timeout=timeout)

    def https_probe(self, url: str, *, timeout: float = 5.0) -> ToolResult:
        parsed = urlparse(url)
        if parsed.scheme != "https":
            url = parsed._replace(scheme="https").geturl() if parsed.scheme else f"https://{url}"
        return self._http_request("https_probe", url, timeout=timeout)

    def path_probe(self, base_url: str, paths: list[str] | None = None, *, timeout: float = 5.0) -> ToolResult:
        paths = paths or ["/", "/robots.txt", "/sitemap.xml", "/login", "/admin"]
        results: list[dict[str, Any]] = []
        errors: list[str] = []
        for path in paths:
            url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
            item = self._http_request("path_probe", url, timeout=timeout)
            status_code = item.output.get("status_code")
            results.append(
                {
                    "path": path,
                    "url": url,
                    "status": item.status,
                    "status_code": status_code,
                    "title": item.output.get("title"),
                }
            )
            if item.error:
                errors.append(f"{path}: {item.error}")
        discovered = [item for item in results if isinstance(item.get("status_code"), int) and item["status_code"] < 500]
        return ToolResult(
            tool_name="path_probe",
            status="success" if discovered else "failed",
            input={"base_url": base_url, "paths": paths, "timeout": timeout},
            output={"results": results, "discovered": discovered},
            error="; ".join(errors) if errors and not discovered else None,
            artifact_type="web_path_probe",
        )

    def host_info(self) -> ToolResult:
        return ToolResult(
            tool_name="host_info",
            status="success",
            input={},
            output={
                "hostname": platform_node(),
                "system": system(),
                "release": release(),
                "platform": platform(),
            },
            artifact_type="host_info",
        )

    def _http_request(self, tool_name: str, url: str, *, timeout: float) -> ToolResult:
        payload = {"url": url, "timeout": timeout}
        try:
            request = Request(url, headers={"User-Agent": "DivinePrototype/0.1"})
            with urlopen(request, timeout=timeout) as response:
                body = response.read(131072)
                headers = {key.lower(): value for key, value in response.headers.items()}
                text = body.decode(_charset(headers), errors="replace")
                return ToolResult(
                    tool_name=tool_name,
                    status="success",
                    input=payload,
                    output={
                        "url": response.geturl(),
                        "status_code": response.status,
                        "headers": headers,
                        "title": _extract_title(text),
                        "body_preview": text[:2000],
                        "content_length": len(body),
                    },
                    artifact_type="http_response",
                )
        except HTTPError as exc:
            headers = {key.lower(): value for key, value in exc.headers.items()}
            body = exc.read(65536)
            text = body.decode(_charset(headers), errors="replace")
            return ToolResult(
                tool_name=tool_name,
                status="success",
                input=payload,
                output={
                    "url": exc.url,
                    "status_code": exc.code,
                    "headers": headers,
                    "title": _extract_title(text),
                    "body_preview": text[:2000],
                    "content_length": len(body),
                },
                artifact_type="http_response",
            )
        except (OSError, URLError) as exc:
            return ToolResult(
                tool_name=tool_name,
                status="failed",
                input=payload,
                output={},
                error=str(exc),
                artifact_type="http_response",
            )


class _TitleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_title = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "title":
            self.in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.parts.append(data.strip())


def _extract_title(html: str) -> str | None:
    parser = _TitleParser()
    try:
        parser.feed(html)
    except Exception:
        return None
    title = " ".join(part for part in parser.parts if part).strip()
    return title or None


def _charset(headers: Mapping[str, str]) -> str:
    content_type = headers.get("content-type", "")
    for part in content_type.split(";"):
        key, _, value = part.strip().partition("=")
        if key.lower() == "charset" and value:
            return value
    return "utf-8"
