from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ProviderConfig:
    """单个 LLM provider 配置"""
    api_key: str = ""
    base_url: str = ""
    timeout: float = 120.0


@dataclass
class LLMConfig:
    """LLM 总配置"""
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    pricing: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class DivineConfig:
    """全局配置"""
    targets: list[str] = field(default_factory=list)
    goal: str = ""
    llm: LLMConfig = field(default_factory=LLMConfig)
    max_rounds: int = 20
    max_tasks: int = 50
    timeout: int = 3600
    concurrency: int = 3
    code_execution_timeout: int = 60
    planner_model: str = "claude-sonnet-4-20250514"
    reflector_model: str = "claude-sonnet-4-20250514"
    executor_model: str = "claude-sonnet-4-20250514"
    db_path: str = ":memory:"
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: Path) -> "DivineConfig":
        """从 YAML 文件加载配置"""
        with open(path) as f:
            raw = yaml.safe_load(f)

        llm_raw = raw.pop("llm", {})
        providers = {}
        for name, pconf in llm_raw.get("providers", {}).items():
            providers[name] = ProviderConfig(**pconf)
        llm = LLMConfig(
            providers=providers,
            pricing=llm_raw.get("pricing", {}),
        )

        return cls(llm=llm, **raw)
