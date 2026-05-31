"""Loguru configuration helpers."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from loguru import logger


DEFAULT_LOGGING_CONFIG_PATH = Path("config/logging.json")

DEFAULT_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "{extra[component]} | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


@dataclass(frozen=True)
class LLMTraceSettings:
    enabled: bool = True
    artifact_dir: str = "artifacts/llm"
    index_path: str = "logs/llm_traces.jsonl"
    save_full_prompt: bool = True
    save_full_response: bool = True
    save_on_success: bool = True
    save_on_failure: bool = True
    variable_preview_chars: int = 300
    large_value_threshold_chars: int = 2000
    artifact_retention_days: int | None = 14
    index_max_bytes: int | None = 20 * 1024 * 1024

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "LLMTraceSettings":
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError("'llm_trace' must be an object in logging config.")
        return cls(
            enabled=_bool(data.get("enabled"), cls.enabled),
            artifact_dir=str(data.get("artifact_dir", cls.artifact_dir)),
            index_path=str(data.get("index_path", cls.index_path)),
            save_full_prompt=_bool(data.get("save_full_prompt"), cls.save_full_prompt),
            save_full_response=_bool(data.get("save_full_response"), cls.save_full_response),
            save_on_success=_bool(data.get("save_on_success"), cls.save_on_success),
            save_on_failure=_bool(data.get("save_on_failure"), cls.save_on_failure),
            variable_preview_chars=_optional_int(
                data.get("variable_preview_chars"),
                cls.variable_preview_chars,
            ),
            large_value_threshold_chars=_optional_int(
                data.get("large_value_threshold_chars"),
                cls.large_value_threshold_chars,
            ),
            artifact_retention_days=_optional_nullable_int(
                data.get("artifact_retention_days"),
                cls.artifact_retention_days,
            ),
            index_max_bytes=_optional_nullable_int(
                data.get("index_max_bytes"),
                cls.index_max_bytes,
            ),
        )


@dataclass(frozen=True)
class LoggingSettings:
    level: str = "INFO"
    console_enabled: bool = True
    console_colorize: bool = True
    file_enabled: bool = True
    file_path: str = "logs/divine.log"
    rotation: str = "20 MB"
    retention: str = "14 days"
    compression: str | None = "zip"
    enqueue: bool = True
    backtrace: bool = False
    diagnose: bool = False
    serialize: bool = False
    fmt: str = DEFAULT_LOG_FORMAT
    llm_trace: LLMTraceSettings = field(default_factory=LLMTraceSettings)

    @classmethod
    def from_file(cls, path: str | Path = DEFAULT_LOGGING_CONFIG_PATH) -> "LoggingSettings":
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        data = _load_json_mapping(config_path)
        return cls(
            level=str(data.get("level", cls.level)).upper(),
            console_enabled=_bool(data.get("console_enabled"), cls.console_enabled),
            console_colorize=_bool(data.get("console_colorize"), cls.console_colorize),
            file_enabled=_bool(data.get("file_enabled"), cls.file_enabled),
            file_path=str(data.get("file_path", cls.file_path)),
            rotation=str(data.get("rotation", cls.rotation)),
            retention=str(data.get("retention", cls.retention)),
            compression=_optional_str(data.get("compression", cls.compression)),
            enqueue=_bool(data.get("enqueue"), cls.enqueue),
            backtrace=_bool(data.get("backtrace"), cls.backtrace),
            diagnose=_bool(data.get("diagnose"), cls.diagnose),
            serialize=_bool(data.get("serialize"), cls.serialize),
            fmt=str(data.get("format", cls.fmt)),
            llm_trace=LLMTraceSettings.from_mapping(data.get("llm_trace")),
        )


def configure_logging(
    settings: LoggingSettings | None = None,
    *,
    config_path: str | Path = DEFAULT_LOGGING_CONFIG_PATH,
) -> LoggingSettings:
    """Configure Loguru sinks and return the resolved settings."""

    resolved = settings or LoggingSettings.from_file(config_path)
    set_logging_settings(resolved)
    logger.remove()
    logger.configure(extra={"component": "app"})

    if resolved.console_enabled:
        logger.add(
            sys.stderr,
            level=resolved.level,
            format=resolved.fmt,
            colorize=resolved.console_colorize,
            enqueue=resolved.enqueue,
            backtrace=resolved.backtrace,
            diagnose=resolved.diagnose,
            serialize=resolved.serialize,
        )

    if resolved.file_enabled:
        log_path = Path(resolved.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_path,
            level=resolved.level,
            format=resolved.fmt,
            rotation=resolved.rotation,
            retention=resolved.retention,
            compression=resolved.compression,
            enqueue=resolved.enqueue,
            backtrace=resolved.backtrace,
            diagnose=resolved.diagnose,
            serialize=resolved.serialize,
        )

    return resolved


_CURRENT_LOGGING_SETTINGS = LoggingSettings()


def set_logging_settings(settings: LoggingSettings) -> None:
    global _CURRENT_LOGGING_SETTINGS
    _CURRENT_LOGGING_SETTINGS = settings


def get_logging_settings() -> LoggingSettings:
    return _CURRENT_LOGGING_SETTINGS


def get_logger(component: str):
    """Return a logger bound to a framework component name."""

    return logger.bind(component=component)


def _load_json_mapping(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in logging config: {path}") from exc

    if not isinstance(data, dict):
        raise ValueError("Logging config root must be a JSON object.")
    return data


def _bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _optional_nullable_int(value: object, default: int | None) -> int | None:
    if value is None:
        return default
    if value == "":
        return None
    return int(value)
