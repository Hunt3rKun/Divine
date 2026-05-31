"""LLM trace artifact and JSONL index writer."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping
from uuid import uuid4

from divine.logger.config import LLMTraceSettings, get_logging_settings
from divine.logger.redaction import redact_data, summarize_messages

if TYPE_CHECKING:
    from divine.llm.types import LLMRequest, LLMResponse


def generate_trace_id(prefix: str = "llm") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}_{uuid4().hex[:8]}"


@dataclass
class LLMTraceContext:
    trace_id: str
    provider: str
    model: str
    agent: str | None
    template_id: str | None
    metadata: Mapping[str, Any]
    started_at: str
    started_perf: float


class LLMTraceRecorder:
    """Writes compact JSONL indexes and full per-call artifacts."""

    def __init__(self, settings: LLMTraceSettings | None = None) -> None:
        self.settings = settings or get_logging_settings().llm_trace
        self._cleanup_done = False

    def start_request(self, request: LLMRequest, *, provider: str, model: str) -> LLMTraceContext:
        trace_id = request.trace_id or generate_trace_id()
        prompt_trace = _mapping_or_none(request.prompt_trace)
        return LLMTraceContext(
            trace_id=trace_id,
            provider=provider,
            model=model,
            agent=request.agent,
            template_id=_string_or_none(prompt_trace.get("template_id")) if prompt_trace else None,
            metadata=dict(request.trace_metadata or {}),
            started_at=_now_iso(),
            started_perf=time.perf_counter(),
        )

    def record_prompt_render(self, prompt_trace: Mapping[str, Any]) -> str | None:
        if not self.settings.enabled:
            return None

        trace_id = str(prompt_trace["trace_id"])
        artifact_path = self._artifact_path(trace_id, suffix="prompt")
        payload = {
            "event_type": "prompt_render",
            "recorded_at": _now_iso(),
            "trace_id": trace_id,
            "prompt_trace": redact_data(prompt_trace),
        }
        self._write_json(artifact_path, payload)
        self._append_index(
            {
                "event_type": "prompt_render",
                "recorded_at": payload["recorded_at"],
                "trace_id": trace_id,
                "agent": prompt_trace.get("agent"),
                "template_id": prompt_trace.get("template_id"),
                "template_version": prompt_trace.get("template_version"),
                "artifact_path": str(artifact_path),
            }
        )
        return str(artifact_path)

    def record_success(self, context: LLMTraceContext, request: LLMRequest, response: LLMResponse) -> str | None:
        if not self.settings.enabled or not self.settings.save_on_success:
            return None

        ended_at = _now_iso()
        artifact_path = self._artifact_path(context.trace_id)
        payload = self._base_payload(context, request, ended_at, "success")
        payload["response"] = self._response_payload(response)
        payload["usage"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "cached_tokens": response.usage.cached_tokens,
            "cache_miss_tokens": response.usage.cache_miss_tokens,
            "reasoning_tokens": response.usage.reasoning_tokens,
        }
        self._write_json(artifact_path, payload)
        self._append_index(self._index_payload(context, payload, artifact_path, response=response))
        return str(artifact_path)

    def record_failure(self, context: LLMTraceContext, request: LLMRequest, error: Exception) -> str | None:
        if not self.settings.enabled or not self.settings.save_on_failure:
            return None

        ended_at = _now_iso()
        artifact_path = self._artifact_path(context.trace_id, suffix="error")
        payload = self._base_payload(context, request, ended_at, "failure")
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        self._write_json(artifact_path, payload)
        self._append_index(self._index_payload(context, payload, artifact_path, error=error))
        return str(artifact_path)

    def _base_payload(
        self,
        context: LLMTraceContext,
        request: LLMRequest,
        ended_at: str,
        status: str,
    ) -> dict[str, Any]:
        messages = request.normalized_messages()
        artifact_messages = _messages_for_artifact(request, messages)
        payload: dict[str, Any] = {
            "event_type": "llm_request",
            "status": status,
            "trace_id": context.trace_id,
            "agent": context.agent,
            "provider": context.provider,
            "model": context.model,
            "metadata": redact_data(dict(context.metadata)),
            "started_at": context.started_at,
            "ended_at": ended_at,
            "latency_ms": round((time.perf_counter() - context.started_perf) * 1000, 3),
            "request": {
                "system": request.system if self.settings.save_full_prompt else None,
                "messages": redact_data(artifact_messages) if self.settings.save_full_prompt else None,
                "message_summaries": summarize_messages(
                    artifact_messages,
                    preview_chars=self.settings.variable_preview_chars,
                ),
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "extra": redact_data(dict(request.extra)),
            },
            "prompt_trace": redact_data(request.prompt_trace) if request.prompt_trace else None,
        }
        return payload

    def _response_payload(self, response: LLMResponse) -> dict[str, Any]:
        content = response.content or ""
        return {
            "content": content if self.settings.save_full_response else None,
            "content_length": len(content),
            "finish_reason": response.finish_reason,
            "reasoning_content": response.reasoning_content if self.settings.save_full_response else None,
        }

    def _index_payload(
        self,
        context: LLMTraceContext,
        payload: Mapping[str, Any],
        artifact_path: Path,
        *,
        response: LLMResponse | None = None,
        error: Exception | None = None,
    ) -> dict[str, Any]:
        item: dict[str, Any] = {
            "event_type": "llm_request",
            "status": payload["status"],
            "recorded_at": payload["ended_at"],
            "trace_id": context.trace_id,
            "agent": context.agent,
            "provider": context.provider,
            "model": context.model,
            "template_id": context.template_id,
            "task_id": context.metadata.get("task_id"),
            "node_id": context.metadata.get("node_id"),
            "execution_id": context.metadata.get("execution_id"),
            "iteration": context.metadata.get("iteration"),
            "turn": context.metadata.get("turn"),
            "phase": context.metadata.get("phase"),
            "latency_ms": payload["latency_ms"],
            "artifact_path": str(artifact_path),
        }
        if response:
            item.update(
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "cached_tokens": response.usage.cached_tokens,
                    "cache_miss_tokens": response.usage.cache_miss_tokens,
                    "finish_reason": response.finish_reason,
                }
            )
        if error:
            item.update({"error_type": type(error).__name__, "error_message": str(error)})
        return item

    def _artifact_path(self, trace_id: str, suffix: str | None = None) -> Path:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filename = f"{trace_id}{'_' + suffix if suffix else ''}.json"
        return Path(self.settings.artifact_dir) / date / filename

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        self._cleanup_old_artifacts()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, default=str)
            file.write("\n")

    def _append_index(self, item: Mapping[str, Any]) -> None:
        index_path = Path(self.settings.index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self._rotate_index_if_needed(index_path)
        with index_path.open("a", encoding="utf-8") as file:
            json.dump(item, file, ensure_ascii=False, default=str)
            file.write("\n")

    def _cleanup_old_artifacts(self) -> None:
        if self._cleanup_done:
            return
        self._cleanup_done = True
        retention_days = self.settings.artifact_retention_days
        if retention_days is None or retention_days <= 0:
            return
        artifact_root = Path(self.settings.artifact_dir)
        if not artifact_root.exists():
            return
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=retention_days)
        for child in artifact_root.iterdir():
            if not child.is_dir():
                continue
            try:
                child_date = datetime.strptime(child.name, "%Y-%m-%d").date()
            except ValueError:
                continue
            if child_date < cutoff:
                for path in child.glob("*.json"):
                    path.unlink(missing_ok=True)
                try:
                    child.rmdir()
                except OSError:
                    pass

    def _rotate_index_if_needed(self, index_path: Path) -> None:
        max_bytes = self.settings.index_max_bytes
        if max_bytes is None or max_bytes <= 0 or not index_path.exists():
            return
        if index_path.stat().st_size < max_bytes:
            return
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        rotated = index_path.with_name(f"{index_path.name}.{timestamp}")
        index_path.rename(rotated)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _messages_for_artifact(request: "LLMRequest", messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prompt_trace = _mapping_or_none(request.prompt_trace)
    rendered_prompt = prompt_trace.get("rendered_prompt") if prompt_trace else None
    if isinstance(rendered_prompt, str) and len(messages) == 1:
        item = dict(messages[0])
        item["content"] = rendered_prompt
        return [item]
    return messages
