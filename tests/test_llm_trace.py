import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from jinja2 import UndefinedError

from divine.context import ContextSection
from divine.llm.config import LLMSettings
from divine.llm.types import LLMRequest, LLMResponse, Message, TokenUsage
from divine.logger import LoggingSettings, configure_logging
from divine.logger.config import LLMTraceSettings
from divine.logger.trace import LLMTraceRecorder
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
        index_lines = (root / "logs" / "llm_traces.jsonl").read_text(encoding="utf-8").splitlines()
        index = json.loads(index_lines[0])
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


def test_trace_recorder_writes_success_artifact_and_index():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        trace_settings = LLMTraceSettings(
            artifact_dir=str(root / "artifacts"),
            index_path=str(root / "logs" / "llm_traces.jsonl"),
            save_full_prompt=True,
            save_full_response=True,
        )
        recorder = LLMTraceRecorder(trace_settings)
        request = LLMRequest(
            messages=[Message("user", "hello")],
            trace_id="llm_test_trace",
            agent="planner",
            prompt_trace={"template_id": "planner.initial_dag", "variables": {"target": "local"}},
            trace_metadata={
                "task_id": "task-1",
                "node_id": "node-1",
                "execution_id": "exec-1",
                "turn": 2,
                "phase": "executor_action",
            },
        )
        response = LLMResponse(
            provider="fake",
            model="fake-model",
            content="ok",
            usage=TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            finish_reason="stop",
        )

        context = recorder.start_request(request, provider="fake", model="fake-model")
        artifact_path = recorder.record_success(context, request, response)

        artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        index = json.loads((root / "logs" / "llm_traces.jsonl").read_text(encoding="utf-8").splitlines()[0])

        assert artifact["trace_id"] == "llm_test_trace"
        assert artifact["request"]["messages"][0]["content"] == "hello"
        assert artifact["response"]["content"] == "ok"
        assert artifact["usage"]["total_tokens"] == 5
        assert artifact["metadata"]["task_id"] == "task-1"
        assert index["artifact_path"] == artifact_path
        assert index["total_tokens"] == 5
        assert index["task_id"] == "task-1"
        assert index["node_id"] == "node-1"
        assert index["execution_id"] == "exec-1"
        assert index["turn"] == 2
        assert index["phase"] == "executor_action"


def test_llm_client_records_trace_with_fake_provider(monkeypatch):
    from divine.llm import client as llm_client_module
    from divine.llm.client import LLMClient

    class FakeProvider:
        def __init__(self, settings):
            self.settings = settings

        def generate(self, request):
            return LLMResponse(
                provider=self.settings.provider,
                model=request.model or self.settings.model,
                content="fake response",
                usage=TokenUsage(prompt_tokens=4, completion_tokens=6, total_tokens=10),
                finish_reason="stop",
            )

    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        trace_settings = LLMTraceSettings(
            artifact_dir=str(root / "artifacts"),
            index_path=str(root / "logs" / "llm_traces.jsonl"),
        )
        configure_logging(
            LoggingSettings(
                console_enabled=False,
                file_enabled=False,
                llm_trace=trace_settings,
            )
        )
        monkeypatch.setitem(llm_client_module.PROVIDER_FACTORIES, "fake", FakeProvider)

        client = LLMClient(LLMSettings(provider="fake", model="fake-model", api_key="test-key"))
        response = client.generate(
            LLMRequest(
                messages=[Message("user", "hello")],
                trace_id="llm_client_trace",
                agent="planner",
                prompt_trace={"template_id": "planner.initial_dag"},
            )
        )

        assert response.content == "fake response"
        index = json.loads((root / "logs" / "llm_traces.jsonl").read_text(encoding="utf-8").splitlines()[0])
        artifact = json.loads(Path(index["artifact_path"]).read_text(encoding="utf-8"))
        assert index["trace_id"] == "llm_client_trace"
        assert artifact["prompt_trace"]["template_id"] == "planner.initial_dag"
        assert artifact["usage"]["total_tokens"] == 10
