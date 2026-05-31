"""Provider-specific prompt cache hint planning."""

from __future__ import annotations

from dataclasses import dataclass

from divine.context.segments import PromptSegment
from divine.logger.redaction import sha256_text


@dataclass(frozen=True)
class CachePolicy:
    enabled: bool = True
    strategy: str = "provider_default"
    ttl: str | None = None
    retention: str | None = None
    key_prefix: str = "divine"


@dataclass(frozen=True)
class CacheHint:
    provider: str
    cache_key: str | None = None
    extra: dict[str, object] | None = None
    anthropic_system_blocks: list[dict[str, object]] | None = None


def build_cache_hint(
    *,
    provider: str,
    agent: str | None,
    stable_segments: list[PromptSegment],
    policy: CachePolicy,
) -> CacheHint:
    if not policy.enabled or not stable_segments:
        return CacheHint(provider=provider, extra={})

    stable_text = "\n\n".join(segment.render() for segment in stable_segments)
    stable_hash = sha256_text(stable_text)[:16]
    agent_part = agent or "agent"
    cache_key = f"{policy.key_prefix}:{agent_part}:{stable_hash}"

    if provider == "openai":
        extra: dict[str, object] = {"prompt_cache_key": cache_key}
        if policy.retention:
            extra["prompt_cache_retention"] = policy.retention
        return CacheHint(provider=provider, cache_key=cache_key, extra=extra)

    if provider == "anthropic":
        blocks: list[dict[str, object]] = []
        cacheable_indexes = [
            index
            for index, segment in enumerate(stable_segments)
            if segment.cache_policy in {"explicit", "automatic"}
        ]
        breakpoint_index = cacheable_indexes[-1] if cacheable_indexes else len(stable_segments) - 1
        for index, segment in enumerate(stable_segments):
            block: dict[str, object] = {"type": "text", "text": segment.render()}
            if index == breakpoint_index:
                cache_control: dict[str, object] = {"type": "ephemeral"}
                if policy.ttl:
                    cache_control["ttl"] = policy.ttl
                block["cache_control"] = cache_control
            blocks.append(block)

        extra = {"cache_control": {"type": "ephemeral"}} if policy.strategy == "automatic" else {}
        return CacheHint(
            provider=provider,
            cache_key=cache_key,
            extra=extra,
            anthropic_system_blocks=blocks,
        )

    return CacheHint(provider=provider, cache_key=cache_key, extra={})
