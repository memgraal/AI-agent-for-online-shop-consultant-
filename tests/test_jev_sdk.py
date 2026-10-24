from types import SimpleNamespace

import pytest
from typesafe_sdk import Choice, Noul

from app.services import jev_classification


class FakeJevClient:
    def __init__(self, response):
        self.response = response
        self.state = None
        self.questions = None
        self.model = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc_info):
        return None

    async def system_one(self, *, state, questions, model=None):
        self.state = state
        self.questions = questions
        self.model = model
        return self.response


@pytest.mark.asyncio
async def test_jev_faq_classification_uses_choice_and_validates_assignments(monkeypatch):
    client = FakeJevClient(SimpleNamespace(choices={
        "question_1": SimpleNamespace(choice="доставка и оплата"),
        "question_2": SimpleNamespace(choice="гарантия и обслуживание"),
    }))
    monkeypatch.setattr(jev_classification, "create_jev_client", lambda: client)

    result = await jev_classification.classify_faq_questions([
        "Где мой заказ?", "Как работает гарантия?",
    ])

    assert [item.category for item in result.assignments] == [
        "доставка и оплата", "гарантия и обслуживание",
    ]
    assert [item.source_index for item in result.assignments] == [1, 2]
    assert all(isinstance(question, Choice) for question in client.questions.values())
    assert "Где мой заказ?" in client.state


@pytest.mark.asyncio
async def test_jev_product_classification_uses_choices_and_nouls_for_tags(monkeypatch):
    client = FakeJevClient(SimpleNamespace(
        choices={
            "category": SimpleNamespace(choice="видеонаблюдение"),
            "product_type": SimpleNamespace(choice="камера"),
            "purpose": SimpleNamespace(choice="видеонаблюдение"),
        },
        nouls={
            "tag_1": SimpleNamespace(noul=0.9),
            "tag_2": SimpleNamespace(noul=0.2),
            "tag_3": SimpleNamespace(noul=0.8),
            "tag_4": SimpleNamespace(noul=0.7),
        },
    ))
    monkeypatch.setattr(jev_classification, "create_jev_client", lambda: client)

    result = await jev_classification.classify_product(
        name="Камера X-100",
        manufacturer="Русичи",
        attributes="Разрешение: 4 Мп; корпус: IP66",
    )

    assert result.category == "видеонаблюдение"
    assert result.product_type == "камера"
    assert result.purpose == "видеонаблюдение"
    assert result.tags == ["Камера X-100", "Разрешение: 4 Мп", "корпус: IP66"]
    assert isinstance(client.questions["category"], Choice)
    assert isinstance(client.questions["tag_1"], Noul)
