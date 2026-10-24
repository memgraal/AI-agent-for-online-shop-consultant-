import pytest
from langchain_core.exceptions import OutputParserException

from app.schemas.admin_catalog import (
    AttributeNormalizationRequest,
    AttributeNormalizationResponse,
    DuplicateScanRequest,
)
from app.services import admin_catalog


class FakeStructuredModel:
    def __init__(self, result):
        self.result = result
        self.schema = None
        self.method = None
        self.calls = []

    def with_structured_output(self, schema, *, method):
        self.schema = schema
        self.method = method
        return self

    async def ainvoke(self, messages, *, max_tokens):
        self.calls.append((messages, max_tokens))
        result = self.result.pop(0) if isinstance(self.result, list) else self.result
        if isinstance(result, BaseException):
            raise result
        return result


async def test_normalization_returns_validated_proposals_and_preserves_source(monkeypatch):
    model = FakeStructuredModel(AttributeNormalizationResponse(attributes=[
        {
            "source_name": "Питание",
            "source_value": "12V DC",
            "normalized_name": "Питание",
            "normalized_value": "12 В постоянного тока",
            "explanation": "Единицы приведены к русскому формату.",
            "confidence": 0.99,
        },
    ]))
    monkeypatch.setattr(admin_catalog, "create_chat_model", lambda: model)
    monkeypatch.setattr(admin_catalog.settings, "llm_provider", "yandex")
    request = AttributeNormalizationRequest(attributes=[
        {"name": "Питание", "value": "12V DC"},
    ])

    result = await admin_catalog.normalize_product_attributes(request)

    assert model.schema is AttributeNormalizationResponse
    assert model.method == "json_mode"
    assert result.attributes[0].source_value == "12V DC"
    assert result.attributes[0].normalized_value == "12 В постоянного тока"


async def test_normalization_retries_malformed_structured_output(monkeypatch):
    model = FakeStructuredModel([
        OutputParserException("bad object"),
        AttributeNormalizationResponse(attributes=[{
            "source_name": "Питание",
            "source_value": "12V DC",
            "normalized_name": "Питание",
            "normalized_value": "12 В постоянного тока",
            "explanation": "Нормализованный формат.",
            "confidence": 0.95,
        }]),
    ])
    monkeypatch.setattr(admin_catalog, "create_chat_model", lambda: model)
    request = AttributeNormalizationRequest(attributes=[
        {"name": "Питание", "value": "12V DC"},
    ])

    result = await admin_catalog.normalize_product_attributes(request)

    assert len(model.calls) == 2
    assert "в том же порядке" in model.calls[1][0][-1].content
    assert len(result.attributes) == 1


def test_duplicate_scan_detects_exact_sku_and_similar_product():
    result = admin_catalog.find_duplicate_candidates(DuplicateScanRequest(products=[
        {
            "sku": "LIVI-100",
            "name": "Датчик движения Livi",
            "manufacturer": "Стелс",
            "attributes": {"питание": "3 В", "тип": "датчик движения"},
        },
        {
            "sku": "LIVI-100",
            "name": "Датчик движения Livi беспроводной",
            "manufacturer": "Стелс",
            "attributes": {"питание": "3 В"},
        },
        {
            "sku": "OTHER-2",
            "name": "Датчик движения Livi",
            "manufacturer": "Стелс",
            "attributes": {"питание": "3 В"},
        },
    ]))

    assert result.product_count == 3
    assert len(result.candidates) == 2
    assert result.candidates[0].confidence == 100
    assert result.candidates[0].matched_fields == ["sku"]
    assert {(candidate.left_index, candidate.right_index) for candidate in result.candidates} == {
        (0, 1), (0, 2),
    }
    assert all(candidate.rationale for candidate in result.candidates)


def test_duplicate_scan_ignores_unrelated_names_from_same_manufacturer():
    result = admin_catalog.find_duplicate_candidates(DuplicateScanRequest(products=[
        {"name": "Пожарный извещатель", "manufacturer": "Полисервис"},
        {"name": "Блок речевого оповещения", "manufacturer": "Полисервис"},
    ]))

    assert result.candidates == []


def test_duplicate_scan_rejects_single_product_batch():
    with pytest.raises(ValueError):
        DuplicateScanRequest(products=[{"name": "Датчик"}])


def test_duplicate_scan_rejects_batches_that_make_pairwise_work_unbounded():
    with pytest.raises(ValueError):
        DuplicateScanRequest(products=[{"name": f"Товар {index}"} for index in range(101)])
