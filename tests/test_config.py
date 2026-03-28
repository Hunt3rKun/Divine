import tempfile
from pathlib import Path

import yaml

from divine.config import DivineConfig, LLMConfig, ProviderConfig


class TestDivineConfig:
    def test_default_values(self):
        cfg = DivineConfig()
        assert cfg.max_rounds == 20
        assert cfg.concurrency == 3
        assert cfg.timeout == 3600
        assert cfg.db_path == ":memory:"

    def test_from_yaml(self):
        data = {
            "targets": ["192.168.1.1"],
            "goal": "获取目标主机控制权",
            "max_rounds": 10,
            "concurrency": 5,
            "llm": {
                "providers": {
                    "openai": {
                        "api_key": "sk-test",
                        "base_url": "https://api.openai.com/v1",
                    }
                },
                "pricing": {
                    "gpt-4o": {"input": 2.5, "output": 10.0}
                },
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            cfg = DivineConfig.from_yaml(Path(f.name))

        assert cfg.targets == ["192.168.1.1"]
        assert cfg.goal == "获取目标主机控制权"
        assert cfg.max_rounds == 10
        assert cfg.concurrency == 5
        assert "openai" in cfg.llm.providers
        assert cfg.llm.providers["openai"].api_key == "sk-test"
        assert cfg.llm.pricing["gpt-4o"]["input"] == 2.5
