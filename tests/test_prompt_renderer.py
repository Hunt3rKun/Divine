import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from jinja2 import UndefinedError

from divine.context import ContextSection
from divine.logger.config import LLMTraceSettings
from divine.prompts import PromptRenderer


def test_prompt_renderer_records_variable_fill_artifact():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        template_dir = root / "templates"
        (template_dir / "agents" / "planner").mkdir(parents=True)
        (template_dir / "shared").mkdir()
        (template_dir / "shared" / "header.j2").write_text(
            "Header: {{ title }}",
            encoding="utf-8",
        )
        (template_dir / "agents" / "planner" / "initial_dag.j2").write_text(
            "{% include 'shared/header.j2' %}\nTarget: {{ target }}\nAPI key: {{ api_key }}\nContext: {{ context }}",
            encoding="utf-8",
        )
        trace_settings = LLMTraceSettings(
            artifact_dir=str(root / "artifacts"),
            index_path=str(root / "logs" / "llm_traces.jsonl"),
            variable_preview_chars=10,
            large_value_threshold_chars=20,
        )

        rendered = PromptRenderer(template_dir, trace_settings=trace_settings).render(
            template_id="agents.planner.initial_dag",
            template_version="v1",
            agent="planner",
            variables={
                "title": "Planner",
                "target": "local test target",
                "api_key": "secret-key",
                "context": "x" * 50,
            },
        )

        assert "local test target" in rendered.content
        assert rendered.referenced_templates == ("shared/header.j2",)
        index = json.loads((root / "logs" / "llm_traces.jsonl").read_text(encoding="utf-8").splitlines()[0])
        artifact = json.loads(Path(index["artifact_path"]).read_text(encoding="utf-8"))

        assert index["event_type"] == "prompt_render"
        assert artifact["prompt_trace"]["template_id"] == "agents.planner.initial_dag"
        assert artifact["prompt_trace"]["variables"]["api_key"] == "[REDACTED]"
        assert "secret-key" not in artifact["prompt_trace"]["rendered_prompt"]
        assert "[REDACTED]" in artifact["prompt_trace"]["rendered_prompt"]
        assert artifact["prompt_trace"]["variable_summaries"]["context"]["large"] is True
        assert artifact["prompt_trace"]["variable_summaries"]["target"]["preview"] == "local test"


def test_prompt_renderer_raises_on_missing_jinja_variable():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        template_dir = root / "templates"
        template_dir.mkdir()
        (template_dir / "missing.j2").write_text("Hello {{ name }}", encoding="utf-8")

        renderer = PromptRenderer(
            template_dir,
            trace_settings=LLMTraceSettings(
                artifact_dir=str(root / "artifacts"),
                index_path=str(root / "logs" / "llm_traces.jsonl"),
            ),
        )

        with pytest.raises(UndefinedError):
            renderer.render(template_id="missing", variables={})


def test_prompt_renderer_can_create_prompt_segment():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        template_dir = root / "templates"
        template_dir.mkdir()
        (template_dir / "segment.j2").write_text("Role: {{ role }}", encoding="utf-8")

        segment = PromptRenderer(
            template_dir,
            trace_settings=LLMTraceSettings(
                artifact_dir=str(root / "artifacts"),
                index_path=str(root / "logs" / "llm_traces.jsonl"),
            ),
        ).render_segment(
            template_id="segment",
            variables={"role": "planner"},
            section=ContextSection.STATIC,
            stable=True,
            cache_policy="explicit",
            agent="planner",
        )

        assert segment.name == "segment"
        assert segment.content == "Role: planner"
        assert segment.section == ContextSection.STATIC
        assert segment.stable is True
        assert segment.metadata["template_id"] == "segment"

