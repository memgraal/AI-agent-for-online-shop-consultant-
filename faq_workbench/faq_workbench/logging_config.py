"""Structured, allowlisted logging for the independent FAQ API process."""

import contextvars
import json
import logging
from datetime import datetime, timezone
from typing import Any

request_id_context: contextvars.ContextVar[str] = contextvars.ContextVar("faq_request_id", default="-")
SAFE_FIELDS = {
    "event", "request_id", "method", "path", "status_code", "duration_ms",
    "error_type", "question_count", "candidate_count", "duplicate_event", "candidate_id",
}


class FAQJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_context.get(),
        }
        for field in SAFE_FIELDS:
            if field != "request_id" and (value := getattr(record, field, None)) is not None:
                payload[field] = value
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(FAQJsonFormatter())
    root.addHandler(handler)


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    safe_fields = {key: value for key, value in fields.items() if key in SAFE_FIELDS and key != "event"}
    logger.log(level, event, extra={"event": event, **safe_fields})
