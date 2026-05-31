"""Cache-aware LLM request assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import hashlib
from datetime import datetime, timezone
from uuid import uuid4

from divine.context.cache_policy import CacheHint, CachePolicy, build_cache_hint
from divine.context.segments import ContextSection, PromptSegment
from divine.context.token_budget import TokenBudget, estimate_tokens
from divine.context.types import LLMRequest, Message


@dataclass(frozen=True)
class ContextBuildResult:
    request: LLMRequest
    segments: list[PromptSegment]
    cache_hint: CacheHint
    stable_prefix_hash: str
    estimated_input_tokens: int


class ContextBuilder:
    def __init__(
        self,
        *,
        provider: str,
        agent: str,
        cache_policy: CachePolicy | None = None,
        token_budget: TokenBudget | None = None,
    ) -> None:
        self.provider = provider
        self.agent = agent
        self.cache_policy = cache_policy or CachePolicy()
        self.token_budget = token_budget

    def build_request(
        self,
        *,
        static_segments: list[PromptSegment],
        mission_segments: list[PromptSegment],
        working_segments: list[PromptSegment],
        current_segments: list[PromptSegment],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        trace_id: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> ContextBuildResult:
        all_segments = [*static_segments, *mission_segments, *working_segments, *current_segments]
        if self.token_budget:
            all_segments = self.token_budget.trim_dynamic_segments(all_segments)

        stable_segments = [segment for segment in all_segments if segment.is_stable_prefix]
        working = [segment for segment in all_segments if segment.section == ContextSection.WORKING]
        current = [segment for segment in all_segments if segment.section == ContextSection.CURRENT]
        cache_hint = build_cache_hint(
            provider=self.provider,
            agent=self.agent,
            stable_segments=stable_segments,
            policy=self.cache_policy,
        )
        stable_prefix = "\n\n".join(segment.render() for segment in stable_segments)
        dynamic_suffix = _render_dynamic_suffix(working, current)
        request_extra = dict(extra or {})
        request_extra.update(cache_hint.extra or {})
        request_trace_id = trace_id or generate_trace_id("llm")

        request = LLMRequest(
            messages=[Message("user", dynamic_suffix)],
            system=_system_payload_for_provider(self.provider, stable_segments, cache_hint),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=request_extra,
            trace_id=request_trace_id,
            agent=self.agent,
            prompt_trace={
                "trace_id": request_trace_id,
                "agent": self.agent,
                "template_id": "context.cache_aware",
                "template_version": "v1",
                "stable_prefix_hash": sha256_text(stable_prefix),
                "stable_segment_names": [segment.name for segment in stable_segments],
                "working_segment_names": [segment.name for segment in working],
                "current_segment_names": [segment.name for segment in current],
                "cache_key": cache_hint.cache_key,
                "cache_strategy": self.cache_policy.strategy,
            },
        )
        estimated_tokens = estimate_tokens(stable_prefix) + estimate_tokens(dynamic_suffix)
        return ContextBuildResult(
            request=request,
            segments=all_segments,
            cache_hint=cache_hint,
            stable_prefix_hash=sha256_text(stable_prefix),
            estimated_input_tokens=estimated_tokens,
        )


def _system_payload_for_provider(
    provider: str,
    stable_segments: list[PromptSegment],
    cache_hint: CacheHint,
) -> Any:
    if provider == "anthropic" and cache_hint.anthropic_system_blocks:
        return cache_hint.anthropic_system_blocks
    return "\n\n".join(segment.render() for segment in stable_segments)


def _render_dynamic_suffix(working: list[PromptSegment], current: list[PromptSegment]) -> str:
    blocks: list[str] = []
    if working:
        blocks.append("# Working Context\n" + "\n\n".join(segment.render() for segment in working))
    if current:
        blocks.append("# Current Instruction\n" + "\n\n".join(segment.render() for segment in current))
    return "\n\n".join(blocks)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_trace_id(prefix: str = "llm") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}_{uuid4().hex[:8]}"
