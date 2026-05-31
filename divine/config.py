from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class DivineConfig:
    """引擎运行配置，从 targets YAML 加载并映射到 TaskContext。"""

    target: str = ""
    goal: str = ""
    scope: list[str] = field(default_factory=list)
    max_iterations: int = 20
    max_consecutive_failures: int = 5
    task_id: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_config: str = "config/llm.json"
    logging_config: str = "config/logging.json"

    @classmethod
    def from_yaml(cls, path: Path) -> "DivineConfig":
        """从 YAML 文件加载配置。

        兼容 `targets: [...]` 写法：取第一个作为 target，整体作为默认 scope。
        未知字段会被忽略，保持向后兼容。
        """
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        targets = raw.pop("targets", None)
        if targets and not raw.get("target"):
            raw["target"] = targets[0]
            raw.setdefault("scope", list(targets))

        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in raw.items() if key in known})
