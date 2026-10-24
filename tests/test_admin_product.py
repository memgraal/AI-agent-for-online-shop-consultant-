import json

from langchain_core.messages import HumanMessage
from langchain_core.exceptions import OutputParserException

from app.schemas.admin_product import (
    ProductClassificationDraft,
    ProductContentDraft,
    ProductSEODraft,
)
from app.services import admin_product
from app.tools import admin as admin_tools


class FakeStructuredModel:
    def __init__(self, result):
        self.result = result
        self.schema = None
        self.methods = []
        self.messages = None

    def with_structured_output(self, schema, *, method):
        self.schema = schema
        self.methods.append(method)
        return self

    async def ainvoke(self, messages, *, max_tokens):
        self.messages = messages
        if isinstance(self.result, list):
            result = self.result.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        return self.result


async def test_admin_product_uses_jev_for_classification_and_chat_model_for_copy(monkeypatch):
    classification = ProductClassificationDraft(
        category="Видеонаблюдение",
        product_type="IP-камера",
        purpose="Наружное наблюдение",
        tags=["уличная", "IP-камера", "видеонаблюдение"],
    )
    chat_model = FakeStructuredModel(ProductSEODraft(
        description="Краткое описание по характеристикам карточки.",
        h1="IP-камера для наружного наблюдения",
        meta_title="IP-камера — описание и характеристики",
        meta_description="Характеристики IP-камеры для наружного наблюдения.",
        category_text="IP-камеры для построения систем видеонаблюдения.",
    ))
    async def fake_classify_product_with_jev(**kwargs):
        assert kwargs == {
            "name": "Камера X-100",
            "manufacturer": "Русичи",
            "attributes": "Разрешение: 4 Мп; корпус: IP66",
        }
        return classification

    monkeypatch.setattr(admin_product, "classify_product_with_jev", fake_classify_product_with_jev)
    monkeypatch.setattr(admin_product, "create_chat_model", lambda: chat_model)
    monkeypatch.setattr(admin_product.settings, "llm_provider", "yandex")

    draft = await admin_product.generate_product_content(
        name="Камера X-100",
        manufacturer="Русичи",
        attributes="Разрешение: 4 Мп; корпус: IP66",
    )

    assert chat_model.schema is ProductSEODraft
    assert chat_model.methods == ["json_mode"]
    assert isinstance(chat_model.messages[1], HumanMessage)
    text_input = json.loads(chat_model.messages[1].content)
    assert text_input["classification"]["category"] == "Видеонаблюдение"
    assert text_input["classification"]["tags"] == [
        "уличная", "IP-камера", "видеонаблюдение"
    ]
    assert draft.classification.category == "Видеонаблюдение"
    assert draft.seo.h1 == "IP-камера для наружного наблюдения"
    assert draft.related_product_criteria == {
        "manufacturer": "Русичи",
        "product_type": "IP-камера",
        "purpose": "Наружное наблюдение",
    }


async def test_classify_product_returns_pydantic_classification(monkeypatch):
    classification = ProductClassificationDraft(
        category="Видеонаблюдение",
        product_type="IP-камера",
        purpose="Наружное наблюдение",
        tags=["IP66"],
    )
    async def fake_classify_product_with_jev(**kwargs):
        assert kwargs == {
            "name": "Камера X-100",
            "manufacturer": "Русичи",
            "attributes": "Разрешение 4 Мп",
        }
        return classification

    monkeypatch.setattr(admin_product, "classify_product_with_jev", fake_classify_product_with_jev)

    result = await admin_product.classify_product(
        " Камера X-100 ", " Русичи ", " Разрешение 4 Мп "
    )

    assert isinstance(result, ProductClassificationDraft)
    assert result == classification


async def test_admin_seo_tool_returns_product_classification_and_text(monkeypatch):
    draft = ProductContentDraft(
        classification=ProductClassificationDraft(
            category="Видеонаблюдение",
            product_type="IP-камера",
            purpose="Наружное наблюдение",
            tags=["уличная"],
        ),
        seo=ProductSEODraft(
            description="Описание",
            h1="Камера",
            meta_title="Камера — характеристики",
            meta_description="Описание камеры.",
            category_text="Камеры для видеонаблюдения.",
        ),
        related_product_criteria={"manufacturer": "Русичи"},
    )

    async def fake_generate_product_content(name, manufacturer, attributes):
        assert (name, manufacturer, attributes) == (
            "Камера X-100", "Русичи", "Разрешение: 4 Мп"
        )
        return draft

    monkeypatch.setattr(admin_tools, "generate_product_content", fake_generate_product_content)

    result = json.loads(await admin_tools.generate_product_seo.ainvoke({
        "name": "Камера X-100",
        "manufacturer": "Русичи",
        "attributes": "Разрешение: 4 Мп",
    }))

    assert result["classification"]["category"] == "Видеонаблюдение"
    assert result["seo"]["h1"] == "Камера"


def test_product_seo_schema_accepts_yandex_field_names_and_serializes_api_names():
    draft = ProductSEODraft.model_validate({
        "description_draft": "Описание товара",
        "h1": "Камера X-100",
        "meta_title": "Камера X-100 — характеристики",
        "meta_description": "Камера X-100 с разрешением 4 Мп.",
        "category_page_text": "Камеры для видеонаблюдения.",
    })

    assert draft.description == "Описание товара"
    assert draft.category_text == "Камеры для видеонаблюдения."
    assert draft.model_dump() == {
        "description": "Описание товара",
        "h1": "Камера X-100",
        "meta_title": "Камера X-100 — характеристики",
        "meta_description": "Камера X-100 с разрешением 4 Мп.",
        "category_text": "Камеры для видеонаблюдения.",
    }


async def test_admin_product_retries_incomplete_structured_seo_response(monkeypatch):
    classification = ProductClassificationDraft(
        category="Видеонаблюдение",
        product_type="IP-камера",
        purpose="Наружное наблюдение",
        tags=["уличная"],
    )
    seo = ProductSEODraft(
        description="Описание камеры.",
        h1="Камера X-100",
        meta_title="Камера X-100 — характеристики",
        meta_description="Камера с разрешением 4 Мп.",
        category_text="Камеры для видеонаблюдения.",
    )
    chat_model = FakeStructuredModel([
        OutputParserException("incomplete JSON"),
        seo,
    ])
    async def fake_classify_product_with_jev(**kwargs):
        return classification

    monkeypatch.setattr(admin_product, "classify_product_with_jev", fake_classify_product_with_jev)
    monkeypatch.setattr(admin_product, "create_chat_model", lambda: chat_model)
    monkeypatch.setattr(admin_product.settings, "llm_provider", "yandex")

    result = await admin_product.generate_product_content(
        "Камера X-100", "Русичи", "Разрешение 4 Мп"
    )

    assert result.seo == seo
    assert len(chat_model.result) == 0
    assert "всеми пятью непустыми строковыми полями" in chat_model.messages[-1].content
