"""Prompt template renderer with variable-fill trace recording."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound, meta

from divine.context.segments import ContextSection, PromptSegment
from divine.logger.config import LLMTraceSettings, get_logging_settings
from divine.logger.redaction import redact_data, sha256_text, summarize_mapping
from divine.logger.trace import LLMTraceRecorder, generate_trace_id
from divine.llm.types import LLMRequest, Message


DEFAULT_TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass(frozen=True)
class RenderedPrompt:
    trace_id: str
    agent: str | None
    template_id: str
    template_version: str
    template_path: str
    variables: Mapping[str, Any]
    variable_summaries: Mapping[str, Mapping[str, Any]]
    content: str
    redacted_content: str
    referenced_templates: tuple[str, ...] = ()

    def as_trace(self, *, include_full_prompt: bool = True) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "agent": self.agent,
            "template_id": self.template_id,
            "template_version": self.template_version,
            "template_path": self.template_path,
            "template_sha256": sha256_text(Path(self.template_path).read_text(encoding="utf-8")),
            "referenced_templates": list(self.referenced_templates),
            "rendered_prompt": self.redacted_content if include_full_prompt else None,
            "rendered_prompt_length": len(self.content),
            "rendered_prompt_sha256": sha256_text(self.content),
            "rendered_prompt_redacted": self.redacted_content != self.content,
            "variables": redact_data(dict(self.variables)),
            "variable_summaries": dict(self.variable_summaries),
        }

    def to_message(self, role: str = "user") -> Message:
        return Message(role=role, content=self.content)

    def to_request(
        self,
        *,
        system: str | None = None,
        role: str = "user",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> LLMRequest:
        return LLMRequest(
            messages=[self.to_message(role=role)],
            model=model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra or {},
            trace_id=self.trace_id,
            agent=self.agent,
            prompt_trace=self.as_trace(include_full_prompt=get_logging_settings().llm_trace.save_full_prompt),
        )


class PromptRenderer:
    """Loads templates, renders variables, and records every render."""

    def __init__(
        self,
        template_dir: str | Path = DEFAULT_TEMPLATE_DIR,
        *,
        recorder: LLMTraceRecorder | None = None,
        trace_settings: LLMTraceSettings | None = None,
    ) -> None:
        self.template_dir = Path(template_dir)
        self.recorder = recorder or LLMTraceRecorder(trace_settings)
        self.trace_settings = trace_settings or get_logging_settings().llm_trace
        self.environment = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            undefined=StrictUndefined,
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )

    def render(
        self,
        *,
        template_id: str,
        variables: Mapping[str, Any],
        template_version: str = "v1",
        template_path: str | Path | None = None,
        trace_id: str | None = None,
        agent: str | None = None,
    ) -> RenderedPrompt:
        template_name = self._template_name(template_id, template_path)
        path = self.template_dir / template_name
        template = self.environment.get_template(template_name)
        content = template.render(**dict(variables))
        redacted_variables = redact_data(dict(variables))
        redacted_content = template.render(**redacted_variables)
        rendered = RenderedPrompt(
            trace_id=trace_id or generate_trace_id("llm"),
            agent=agent,
            template_id=template_id,
            template_version=template_version,
            template_path=str(path),
            variables=dict(variables),
            variable_summaries=summarize_mapping(
                variables,
                preview_chars=self.trace_settings.variable_preview_chars,
                large_value_threshold_chars=self.trace_settings.large_value_threshold_chars,
            ),
            content=content,
            redacted_content=redacted_content,
            referenced_templates=tuple(sorted(self._referenced_templates(template_name))),
        )
        self.recorder.record_prompt_render(
            rendered.as_trace(include_full_prompt=self.trace_settings.save_full_prompt)
        )
        return rendered

    def render_segment(
        self,
        *,
        template_id: str,
        variables: Mapping[str, Any] | None = None,
        section: ContextSection,
        name: str | None = None,
        template_version: str = "v1",
        stable: bool = False,
        cache_policy: str = "none",
        metadata: Mapping[str, object] | None = None,
        trace_id: str | None = None,
        agent: str | None = None,
    ) -> PromptSegment:
        rendered = self.render(
            template_id=template_id,
            variables=variables or {},
            template_version=template_version,
            trace_id=trace_id,
            agent=agent,
        )
        return PromptSegment(
            name=name or template_id,
            content=rendered.content,
            section=section,
            version=template_version,
            stable=stable,
            cache_policy=cache_policy,
            metadata={
                **dict(metadata or {}),
                "trace_id": rendered.trace_id,
                "template_id": rendered.template_id,
                "template_path": rendered.template_path,
            },
        )

    def _template_name(self, template_id: str, template_path: str | Path | None) -> str:
        if template_path:
            path = Path(template_path)
            if path.is_absolute():
                try:
                    return str(path.relative_to(self.template_dir))
                except ValueError as exc:
                    raise ValueError(f"Template path must be under {self.template_dir}: {path}") from exc
            return str(path)

        candidates = [
            f"{template_id}.j2",
            f"{template_id.replace('.', '/')}.j2",
            f"{template_id.replace('.', '_')}.j2",
        ]
        for candidate in candidates:
            if (self.template_dir / candidate).exists():
                return candidate
        raise FileNotFoundError(f"Prompt template not found for id '{template_id}' under {self.template_dir}")

    def _referenced_templates(self, template_name: str) -> set[str]:
        try:
            source = self.environment.loader.get_source(self.environment, template_name)[0]
        except TemplateNotFound:
            return set()
        parsed = self.environment.parse(source)
        refs = {
            ref
            for ref in meta.find_referenced_templates(parsed)
            if isinstance(ref, str)
        }
        nested = set(refs)
        for ref in refs:
            nested.update(self._referenced_templates(ref))
        return nested
