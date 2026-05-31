from textwrap import dedent

from divine.config import DivineConfig


def test_from_yaml_maps_basic_fields(tmp_path):
    path = tmp_path / "targets.yaml"
    path.write_text(
        dedent(
            """
            target: http://127.0.0.1:8080
            goal: Identify reachable services
            max_iterations: 3
            max_consecutive_failures: 2
            llm_provider: anthropic
            """
        ).strip()
    )

    cfg = DivineConfig.from_yaml(path)

    assert cfg.target == "http://127.0.0.1:8080"
    assert cfg.goal == "Identify reachable services"
    assert cfg.max_iterations == 3
    assert cfg.max_consecutive_failures == 2
    assert cfg.llm_provider == "anthropic"
    assert cfg.llm_config == "config/llm.json"


def test_from_yaml_targets_list_fills_target_and_scope(tmp_path):
    path = tmp_path / "targets.yaml"
    path.write_text(
        dedent(
            """
            targets:
              - http://10.0.0.1
              - http://10.0.0.2
            goal: Map the lab
            """
        ).strip()
    )

    cfg = DivineConfig.from_yaml(path)

    assert cfg.target == "http://10.0.0.1"
    assert cfg.scope == ["http://10.0.0.1", "http://10.0.0.2"]


def test_from_yaml_ignores_unknown_fields(tmp_path):
    path = tmp_path / "targets.yaml"
    path.write_text(
        dedent(
            """
            target: http://127.0.0.1
            goal: Smoke
            legacy_field: should_be_ignored
            """
        ).strip()
    )

    cfg = DivineConfig.from_yaml(path)

    assert cfg.target == "http://127.0.0.1"
    assert not hasattr(cfg, "legacy_field")
