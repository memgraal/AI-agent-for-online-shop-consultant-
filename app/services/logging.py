"""Centralized safe structured logging and per-request correlation context."""

import contextvars
import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings

request_id_context: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)

_SAFE_FIELDS = (
    "event", "request_id", "method", "path", "status_code", "duration_ms",
    "agent", "tool", "repository", "operation", "configured", "history_count",
    "message_chars", "answer_chars", "detail", "error_type", "client_host",
    "log_level", "log_format",
    "database",
    "input_tokens", "output_tokens", "total_tokens",
    "llm_provider", "llm_model",
    "category_count", "confidence",
    "revision",
    "tool_names", "selected_tool_count", "available_tool_count",
)


class JsonFormatter(logging.Formatter):
    """Serialize only approved metadata fields; never dump arbitrary LogRecord extras."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_context.get(),
        }
        for field in _SAFE_FIELDS:
            if field == "request_id":
                continue
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__ if record.exc_info[0] else "Exception"
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """Readable local-development format with the same safe metadata allowlist."""

    def format(self, record: logging.LogRecord) -> str:
        fields = [f"request_id={request_id_context.get()}"]
        for name in _SAFE_FIELDS:
            if name == "request_id":
                continue
            value = getattr(record, name, None)
            if value is not None:
                fields.append(f"{name}={value}")
        base = f"{self.formatTime(record)} {record.levelname} {record.name}: {record.getMessage()}"
        if record.exc_info:
            base += f" exception_type={record.exc_info[0].__name__ if record.exc_info[0] else 'Exception'}"
        return f"{base} {' '.join(fields)}"


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit a named event, filtering metadata to the explicitly safe schema."""
    safe = {key: value for key, value in fields.items() if key in _SAFE_FIELDS and key != "event"}
    logger.log(level, event, extra={"event": event, **safe})


def configure_logging() -> None:
    level_name = settings.log_level.upper()
    level = getattr(logging, level_name, logging.INFO)
    formatter: logging.Formatter = JsonFormatter() if settings.log_format.lower() == "json" else TextFormatter()

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if settings.log_file:
        log_path = Path(settings.log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max(1, settings.log_max_bytes),
            backupCount=max(0, settings.log_backup_count),
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    log_event(logging.getLogger(__name__), logging.INFO, "logging.configured",
              log_level=level_name, log_format=settings.log_format.lower())


def close_logging() -> None:
    root = logging.getLogger()
    for handler in root.handlers[:]:
        handler.flush()
        handler.close()
        root.removeHandler(handler)
