import json
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from app.agents import faq_analyzer
from app.agents.faq_schemas import (
    FAQAnalysisDraft,
    FAQClusterDraft,
    FAQTextClusterDraft,
    FAQTextDraft,
    FAQCategoryAssignment,
    JEvClassificationDraft,
)
from app.schemas.faq import FAQQuestionInput
from app.services import faq_analysis


class FakeStructuredModel:
    def __init__(self, result: Any):
        self.result = result
        self.schema = None
        self.method = None
        self.messages = None
        self.max_tokens = None

    def with_structured_output(self, schema, *, method):
        self.schema = schema
        self.method = method
        return self

    async def ainvoke(self, messages, *, max_tokens) -> Any:
        self.messages = messages
        self.max_tokens = max_tokens
        return self.result


@pytest.mark.asyncio
async def test_jev_classifies_categories_and_chat_model_writes_faq_text(monkeypatch):
    jev_draft = JEvClassificationDraft(assignments=[
        FAQCategoryAssignment(source_index=1, category="доставка и оплата"),
        FAQCategoryAssignment(source_index=2, category="гарантия и обслуживание"),
    ])
    text_draft = FAQTextDraft(
        summary="Частые вопросы о доставке",
        clusters=[FAQTextClusterDraft(
            source_indexes=[1, 2],
            normalized_question="Как получить помощь с заказом?",
            faq_score=81,
            rationale="Обобщённый текст подготовлен основной моделью.",
        )],
    )
    chat_model = FakeStructuredModel(text_draft)

    classified_questions = []

    async def fake_classify_faq_questions(questions):
        classified_questions.extend(questions)
        return jev_draft

    monkeypatch.setattr(faq_analyzer, "classify_faq_questions", fake_classify_faq_questions)
    monkeypatch.setattr(faq_analyzer, "create_chat_model", lambda: chat_model)

    result = await faq_analyzer.build_faq_analyzer().ainvoke(
        {
            "input_text": json.dumps([
                {"index": 1, "question": "Где мой заказ?"},
                {"index": 2, "question": "Как работает гарантия?"},
            ], ensure_ascii=False),
            "classification": None,
            "text_analysis": None,
            "analysis": None,
        }
    )

    assert classified_questions == ["Где мой заказ?", "Как работает гарантия?"]
    assert chat_model.schema is FAQTextDraft
    text_input = json.loads(chat_model.messages[1].content)
    assert text_input[0]["category"] == "доставка и оплата"
    assert text_input[1]["category"] == "гарантия и обслуживание"
    assert result["analysis"].summary == "Частые вопросы о доставке"
    assert [cluster.category for cluster in result["analysis"].clusters] == [
        "доставка и оплата",
        "гарантия и обслуживание",
    ]
    assert all(cluster.normalized_question == "Как получить помощь с заказом?"
               for cluster in result["analysis"].clusters)


@pytest.mark.asyncio
async def test_faq_classification_aggregates_jev_groups_and_keeps_unassigned_questions(
    monkeypatch,
):
    class FakeAnalyzer:
        def __init__(self):
            self.request = None

        async def ainvoke(self, request, *, config):
            self.request = request
            return {"analysis": FAQAnalysisDraft(
                summary="Основной поток вопросов — оплата.",
                clusters=[FAQClusterDraft(
                    source_indexes=[1, 2, 2, 99],
                    normalized_question="Как оплатить заказ?",
                    category="доставка и оплата",
                    faq_score=75,
                    rationale="Оба вопроса о способах оплаты.",
                )],
            )}

    analyzer = FakeAnalyzer()
    monkeypatch.setattr(faq_analysis, "_analyzer", lambda: analyzer)
    questions = [
        FAQQuestionInput(
            text="Как оплатить картой? Напишите на client@example.ru",
            frequency=4,
        ),
        FAQQuestionInput(text="Принимаете банковскую карту?", frequency=3),
        FAQQuestionInput(text="Какая гарантия на товар?", frequency=2),
    ]

    response = await faq_analysis.analyze_faq_questions(questions)

    sent_questions = json.loads(analyzer.request["input_text"])
    assert "client@example.ru" not in analyzer.request["input_text"]
    assert sent_questions[0]["question"].endswith("[СКРЫТО]")
    assert response.question_count == 3
    assert response.summary == "Основной поток вопросов — оплата."
    assert response.clusters[0].source_indexes == [1, 2]
    assert response.clusters[0].category == "доставка и оплата"
    assert response.clusters[0].frequency == 7
    assert response.clusters[0].priority == "высокий"
    assert response.clusters[1].source_indexes == [3]
    assert response.clusters[1].category == "прочее"
    assert response.clusters[1].priority == "проверить"
