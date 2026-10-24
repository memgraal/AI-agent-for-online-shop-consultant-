from uuid import uuid4

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from faq_workbench.api import _normalized, _redact, require_faq_admin
from faq_workbench.schemas import QuestionSubmission, validate_ai_analysis_response
from faq_workbench.settings import settings


def test_question_submission_accepts_optional_idempotency_metadata():
    event_id = uuid4()
    payload = QuestionSubmission(
        question="Как оплатить заказ?",
        conversation_id="conversation-1",
        event_id=event_id,
    )

    assert payload.event_id == event_id
    assert payload.conversation_id == "conversation-1"


@pytest.mark.parametrize("question", ["", "ab", "x" * 1_001])
def test_question_submission_rejects_invalid_length(question):
    with pytest.raises(ValidationError):
        QuestionSubmission(question=question)


def test_question_submission_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        QuestionSubmission(question="Как оплатить заказ?", role="admin")


def test_normalization_and_redaction_happen_before_aggregation():
    assert _normalized("  КАК   ОПЛАТИТЬ\nзаказ? ") == "как оплатить заказ?"
    assert _redact("Напишите на user@example.com или +7 999 123-45-67").find("user@example.com") == -1
    assert "[телефон]" in _redact("Звоните +7 999 123-45-67")


@pytest.mark.asyncio
async def test_admin_auth_rejects_unconfigured_and_wrong_keys(monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "")
    with pytest.raises(HTTPException) as unconfigured:
        await require_faq_admin("secret")
    assert unconfigured.value.status_code == 503

    monkeypatch.setattr(settings, "admin_api_key", "expected-secret")
    with pytest.raises(HTTPException) as unauthorized:
        await require_faq_admin("wrong-secret")
    assert unauthorized.value.status_code == 401


@pytest.mark.asyncio
async def test_admin_auth_accepts_matching_key(monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "expected-secret")
    assert await require_faq_admin("expected-secret") is None


def test_ai_analysis_contract_requires_complete_unique_question_coverage():
    valid = {
        "summary": "Сводка вопросов.",
        "question_count": 2,
        "clusters": [
            {
                "source_indexes": [1],
                "normalized_question": "Как оплатить?",
                "category": "доставка и оплата",
                "frequency": 3,
                "faq_score": 70,
                "priority": "высокий",
                "rationale": "Часто задают.",
            },
            {
                "source_indexes": [2],
                "normalized_question": "Как вернуть?",
                "category": "гарантия и обслуживание",
                "frequency": 1,
                "faq_score": 45,
                "priority": "средний",
                "rationale": "Полезно для покупателей.",
            },
        ],
    }

    assert validate_ai_analysis_response(valid, expected_question_count=2).question_count == 2

    duplicated = {**valid, "clusters": [valid["clusters"][0], {**valid["clusters"][1], "source_indexes": [1]}]}
    with pytest.raises(ValueError, match="assign every source exactly once"):
        validate_ai_analysis_response(duplicated, expected_question_count=2)

    with pytest.raises(ValueError, match="unexpected question count"):
        validate_ai_analysis_response(valid, expected_question_count=3)


def test_ai_analysis_contract_rejects_invalid_payload_before_candidate_replacement():
    with pytest.raises(ValueError, match="invalid FAQ analysis response"):
        validate_ai_analysis_response({"error": "upstream failed"}, expected_question_count=1)
