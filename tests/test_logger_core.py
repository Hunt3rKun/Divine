from pathlib import Path
from tempfile import TemporaryDirectory

from divine.logger import LoggingSettings, configure_logging, get_logger


def test_logging_settings_uses_defaults_when_config_missing():
    settings = LoggingSettings.from_file("missing-logging-config.json")

    assert settings.level == "INFO"
    assert settings.file_path == "logs/divine.log"
    assert settings.file_enabled is True


def test_logging_settings_loads_json_config():
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "logging.json"
        path.write_text(
            """{
  "level": "DEBUG",
  "console_enabled": false,
  "file_enabled": true,
  "file_path": "tmp/test.log",
  "enqueue": false,
  "compression": null,
  "llm_trace": {
    "artifact_dir": "tmp/artifacts",
    "index_path": "tmp/llm_traces.jsonl",
    "variable_preview_chars": 42,
    "artifact_retention_days": 7,
    "index_max_bytes": 1024
  }
}
""",
            encoding="utf-8",
        )

        settings = LoggingSettings.from_file(path)

    assert settings.level == "DEBUG"
    assert settings.console_enabled is False
    assert settings.file_path == "tmp/test.log"
    assert settings.enqueue is False
    assert settings.compression is None
    assert settings.llm_trace.artifact_dir == "tmp/artifacts"
    assert settings.llm_trace.index_path == "tmp/llm_traces.jsonl"
    assert settings.llm_trace.variable_preview_chars == 42
    assert settings.llm_trace.artifact_retention_days == 7
    assert settings.llm_trace.index_max_bytes == 1024


def test_configure_logging_writes_file_sink():
    with TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "divine.log"
        settings = LoggingSettings(
            console_enabled=False,
            file_enabled=True,
            file_path=str(log_path),
            enqueue=False,
            compression=None,
        )

        configure_logging(settings)
        get_logger("test").info("hello logging")

        assert log_path.exists()
        assert "hello logging" in log_path.read_text(encoding="utf-8")
