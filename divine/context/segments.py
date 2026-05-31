"""Context segment primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class ContextSection(str, Enum):
    STATIC = "static"
    MISSION = "mission"
    WORKING = "working"
    CURRENT = "current"


@dataclass(frozen=True)
class PromptSegment:
    name: str
    content: str
    section: ContextSection
    version: str = "v1"
    stable: bool = False
    cache_policy: str = "none"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def render(self) -> str:
        return f"## {self.name}@{self.version}\n{self.content.strip()}"

    @property
    def is_stable_prefix(self) -> bool:
        return self.section in {ContextSection.STATIC, ContextSection.MISSION} and self.stable
