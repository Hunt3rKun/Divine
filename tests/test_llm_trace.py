import json
from pathlib import Path
from tempfile import TemporaryDirectory

from divine.context.types import LLMRequest, Message
from divine.llm.base import LLMResponse, TokenUsage
from divine.logger.config import LLMTraceSettings
from divine.logger.trace import LLMTraceRecorder


def test_trace_recorder_writes_success_artifact_and_index():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        recorder = LLMTraceRecorder(
            LLMTraceSettings(
                artifact_dir=str(root / "artifacts"),
                index_path=str(root / "logs" / "llm_traces.jsonl"),
                save_full_prompt=True,
                save_full_response=True,
            )
        )
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
            content="ok",
            model="fake-model",
            usage=TokenUsage(input_tokens=3, output_tokens=2, total_tokens=5),
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
        assert index["prompt_tokens"] == 3
        assert index["completion_tokens"] == 2
        assert index["total_tokens"] == 5
        assert index["task_id"] == "task-1"
        assert index["node_id"] == "node-1"
        assert index["execution_id"] == "exec-1"
        assert index["turn"] == 2
        assert index["phase"] == "executor_action"


def test_trace_recorder_redacts_sensitive_prompt_fields():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        recorder = LLMTraceRecorder(
            LLMTraceSettings(
                artifact_dir=str(root / "artifacts"),
                index_path=str(root / "logs" / "llm_traces.jsonl"),
            )
        )
        request = LLMRequest(
            messages=[Message("user", "hello")],
            trace_id="llm_redact_trace",
            prompt_trace={
                "template_id": "test",
                "api_key": "sk-secret",
                "rendered_prompt": "API key: sk-secret",
            },
        )
        response = LLMResponse(
            content="ok",
            model="fake-model",
            usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        )

        context = recorder.start_request(request, provider="fake", model="fake-model")
        artifact_path = recorder.record_success(context, request, response)
        artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))

        assert artifact["prompt_trace"]["api_key"] == "[REDACTED]"
        assert artifact["request"]["messages"][0]["content"] == "API key: sk-secret"

