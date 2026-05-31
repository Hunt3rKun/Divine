"""Conservative token-budget helpers.

This module intentionally uses a provider-neutral estimate. Provider SDK usage
fields remain the source of truth after calls return.
"""

from __future__ import annotations

from dataclasses import dataclass

from divine.context.segments import ContextSection, PromptSegment


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return max(1, int(ascii_chars * 0.3 + non_ascii_chars * 0.6))


@dataclass(frozen=True)
class TokenBudget:
    max_input_tokens: int
    reserve_output_tokens: int = 4096

    @property
    def available_input_tokens(self) -> int:
        return max(0, self.max_input_tokens - self.reserve_output_tokens)

    def trim_dynamic_segments(self, segments: list[PromptSegment]) -> list[PromptSegment]:
        stable = [segment for segment in segments if segment.section in {ContextSection.STATIC, ContextSection.MISSION}]
        dynamic = [segment for segment in segments if segment.section not in {ContextSection.STATIC, ContextSection.MISSION}]
        used = sum(estimate_tokens(segment.render()) for segment in stable)
        remaining = self.available_input_tokens - used
        if remaining <= 0:
            return stable

        kept_dynamic: list[PromptSegment] = []
        for segment in dynamic:
            rendered = segment.render()
            estimated = estimate_tokens(rendered)
            if estimated <= remaining:
                kept_dynamic.append(segment)
                remaining -= estimated
                continue
            trimmed = _trim_to_estimated_tokens(rendered, remaining)
            if trimmed:
                kept_dynamic.append(
                    PromptSegment(
                        name=f"{segment.name}.trimmed",
                        content=trimmed,
                        section=segment.section,
                        version=segment.version,
                        stable=False,
                        cache_policy="none",
                        metadata={**dict(segment.metadata), "trimmed": True},
                    )
                )
            break
        return [*stable, *kept_dynamic]


def _trim_to_estimated_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    ratio = max_tokens / max(estimate_tokens(text), 1)
    max_chars = max(0, int(len(text) * ratio))
    if max_chars <= 0:
        return ""
    return text[:max_chars].rstrip() + "\n[TRUNCATED]"
