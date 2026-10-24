import json
import logging

from app.services.logging import JsonFormatter, log_event
from app.services.privacy import redact_personal_data


def test_redact_personal_data_masks_email_phone_and_numeric_id():
    value = "Связаться: user@example.ru, +7 (999) 123-45-67, ИНН 123456789012"

    redacted = redact_personal_data(value)

    assert "user@example.ru" not in redacted
    assert "+7 (999) 123-45-67" not in redacted
    assert "123456789012" not in redacted
    assert redacted.count("[СКРЫТО]") == 3


def test_json_formatter_keeps_safe_event_fields_and_drops_arbitrary_fields():
    logger = logging.getLogger("tests.logging")
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        __file__,
        1,
        "chat.completed",
        (),
        None,
        extra={"event": "chat.completed", "answer_chars": 20, "user_text": "secret"},
    )

    payload = json.loads(JsonFormatter().format(record))

    assert payload["event"] == "chat.completed"
    assert payload["answer_chars"] == 20
    assert "user_text" not in payload


def test_log_event_does_not_emit_unapproved_fields(caplog):
    logger = logging.getLogger("tests.safe_logging")

    with caplog.at_level(logging.INFO, logger=logger.name):
        log_event(logger, logging.INFO, "chat.completed", answer_chars=42, user_text="secret")

    assert "chat.completed" in caplog.text
    assert "secret" not in caplog.text
    assert caplog.records[-1].answer_chars == 42
    assert not hasattr(caplog.records[-1], "user_text")
