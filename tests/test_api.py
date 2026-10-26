import json
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from app.api import admin as admin_api
from app.api import chat as customer_api
from app.config import settings
from app.main import app
from app.main import health
from app.repositories.knowledge import KnowledgeRepository, knowledge_repository
from app.services.rate_limit import rate_limiter
from app.schemas.admin_product import (
    ProductClassificationDraft,
    ProductContentDraft,
    ProductSEODraft,
)
from app.schemas.admin_catalog import (
    AttributeNormalizationResponse,
    DuplicateScanResponse,
)
from app.schemas.content_controls import ContentControlsInput
from app.services.content_controls import ContentControlsStore
from app.services.model_capacity import ModelOverloadedError
from sqlalchemy.exc import SQLAlchemyError


class FakeGraph:
    def __init__(self, answer: str):
        self.answer = answer
        self.received = None

    async def ainvoke(self, input_value, config=None):
        self.received = (input_value, config)
        return {"messages": [AIMessage(content=self.answer)]}


class FakeStreamingGraph(FakeGraph):
    async def astream_events(self, input_value, config=None, version="v2"):
        self.received = (input_value, config)
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": AIMessageChunk(content="Подбор готов")},
        }


class _HealthConnection:
    def __init__(self, error=None):
        self.error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def execute(self, _query):
        if self.error is not None:
            raise self.error


class _HealthEngine:
    def __init__(self, error=None):
        self.error = error

    def connect(self):
        return _HealthConnection(self.error)


async def test_health_reports_unconfigured_database(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "")

    assert await health() == {"status": "ok", "database": "not_configured"}


async def test_health_reports_database_connection_failure(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "postgresql+asyncpg://db/test")
    monkeypatch.setattr(app.state, "database_engine", _HealthEngine(SQLAlchemyError("down")), raising=False)

    response = await health()

    assert response.status_code == 503
    assert response.body == b'{"status":"error","database":"unavailable"}'


async def test_health_reports_database_not_initialized(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "postgresql+asyncpg://db/test")
    if hasattr(app.state, "database_engine"):
        delattr(app.state, "database_engine")

    response = await health()

    assert response.status_code == 503
    assert response.body == b'{"status":"error","database":"not_initialized"}'


async def test_health_reports_database_success(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "postgresql+asyncpg://db/test")
    monkeypatch.setattr(app.state, "database_engine", _HealthEngine(), raising=False)

    assert await health() == {"status": "ok", "database": "ok"}


@pytest.fixture
async def client(monkeypatch):
    # Chat endpoints intentionally require a configured browser Origin.
    monkeypatch.setattr(settings, "allowed_origins", "http://test")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Origin": "http://test"},
    ) as test_client:
        yield test_client


async def test_customer_chat_uses_async_graph_and_returns_request_id(client, monkeypatch):
    fake_graph = FakeGraph("Ответ готов")
    monkeypatch.setattr(customer_api, "_agent", lambda: fake_graph)

    response = await client.post(
        "/api/chat",
        headers={"X-Request-ID": "test-request-42"},
        json={"message": "Почта user@example.ru", "detail": "quick"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Ответ готов"
    assert payload["conversation_id"]
    assert response.headers["X-Request-ID"] == "test-request-42"
    input_state, config = fake_graph.received
    assert "user@example.ru" not in input_state["messages"][0].content
    assert config["recursion_limit"] == settings.agent_recursion_limit


async def test_chat_rejects_unknown_fields(client):
    response = await client.post(
        "/api/chat",
        json={"message": "Привет", "max_tokens": 100000},
    )

    assert response.status_code == 422


async def test_chat_returns_retryable_overload_status(client, monkeypatch):
    class OverloadedChatService:
        async def reply(self, _request):
            raise ModelOverloadedError("internal queue state")

    monkeypatch.setattr(customer_api, "_service", lambda: OverloadedChatService())
    response = await client.post("/api/chat", json={"message": "Привет"})

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"


async def test_readiness_reports_missing_provider_configuration(client, monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "llm_model", "gpt-4o-mini")
    monkeypatch.setattr(settings, "llm_api_key", "chat-secret")
    monkeypatch.setattr(settings, "jev_llm_model", "typesafe/jev-1.14")
    monkeypatch.setattr(settings, "jev_llm_base_url", "https://routerai.ru/api")
    monkeypatch.setattr(settings, "jev_llm_api_key", "jev-secret")
    monkeypatch.setattr(settings, "admin_api_key", "admin-secret")

    ready = await client.get("/ready")
    monkeypatch.setattr(settings, "jev_llm_api_key", "")
    not_ready = await client.get("/ready")

    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert not_ready.status_code == 503
    assert not_ready.json()["checks"]["jev_provider"] is False


async def test_product_classification_reports_retryable_overload(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "test-secret")

    async def overloaded(*_args):
        raise ModelOverloadedError("internal queue state")

    monkeypatch.setattr(admin_api, "classify_product", overloaded)
    response = await client.post(
        "/api/admin/assistant/products/classify",
        headers={"X-Admin-Key": "test-secret"},
        json={"name": "Камера", "manufacturer": "SSS", "attributes": "IP66"},
    )

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"
    assert response.json()["detail"] == "AI-сервис перегружен. Повторите запрос позже."


async def test_admin_chat_rejects_missing_key(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "test-secret")

    response = await client.post(
        "/api/admin/assistant/chat",
        json={"message": "Сводка обращений"},
    )

    assert response.status_code == 401


async def test_admin_chat_accepts_configured_key_and_calls_async_graph(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "test-secret")
    fake_graph = FakeGraph("Админский ответ")
    monkeypatch.setattr(admin_api, "_agent", lambda: fake_graph)

    response = await client.post(
        "/api/admin/assistant/chat",
        headers={"X-Admin-Key": "test-secret"},
        json={"message": "Сделай SEO-черновик", "detail": "detailed"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Админский ответ"
    assert payload["conversation_id"]
    assert fake_graph.received[0]["detail"] == "detailed"


async def test_admin_product_seo_draft_returns_fixed_schema(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "test-secret")
    draft = ProductContentDraft(
        classification=ProductClassificationDraft(
            category="Видеонаблюдение",
            product_type="IP-камера",
            purpose="Наружное наблюдение",
            tags=["IP66"],
        ),
        seo=ProductSEODraft(
            description="Описание камеры.",
            h1="Камера X-100",
            meta_title="Камера X-100 — 4 Мп",
            meta_description="Камера X-100 с разрешением 4 Мп.",
            category_text="IP-камеры для видеонаблюдения.",
        ),
        related_product_criteria={"manufacturer": "Русичи"},
    )

    async def fake_generate_product_content(name, manufacturer, attributes):
        assert (name, manufacturer, attributes) == (
            "Камера X-100", "Русичи", "Разрешение 4 Мп"
        )
        return draft

    monkeypatch.setattr(admin_api, "generate_product_content", fake_generate_product_content)

    response = await client.post(
        "/api/admin/assistant/products/seo-draft",
        headers={"X-Admin-Key": "test-secret"},
        json={
            "name": "Камера X-100",
            "manufacturer": "Русичи",
            "attributes": "Разрешение 4 Мп",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["classification"]["category"] == "Видеонаблюдение"
    assert payload["seo"]["description"] == "Описание камеры."
    assert payload["seo"]["category_text"] == "IP-камеры для видеонаблюдения."
    assert set(payload["seo"]) == {
        "description", "h1", "meta_title", "meta_description", "category_text"
    }


async def test_admin_product_classification_returns_pydantic_contract(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "test-secret")
    classification = ProductClassificationDraft(
        category="Видеонаблюдение",
        product_type="IP-камера",
        purpose="Наружное наблюдение",
        tags=["уличная", "IP66"],
    )

    async def fake_classify_product(name, manufacturer, attributes):
        assert (name, manufacturer, attributes) == (
            "Камера X-100", "Русичи", "Разрешение 4 Мп"
        )
        return classification

    monkeypatch.setattr(admin_api, "classify_product", fake_classify_product)

    response = await client.post(
        "/api/admin/assistant/products/classify",
        headers={"X-Admin-Key": "test-secret"},
        json={
            "name": "Камера X-100",
            "manufacturer": "Русичи",
            "attributes": "Разрешение 4 Мп",
        },
    )

    assert response.status_code == 200
    assert response.json() == classification.model_dump()


async def test_admin_attribute_normalization_returns_typed_mapping(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "test-secret")
    expected = AttributeNormalizationResponse(attributes=[{
        "source_name": "Питание",
        "source_value": "12V DC",
        "normalized_name": "Питание",
        "normalized_value": "12 В постоянного тока",
        "explanation": "Единицы приведены к русскому формату.",
        "confidence": 0.99,
    }])

    async def fake_normalize(request):
        assert request.attributes[0].value == "12V DC"
        return expected

    monkeypatch.setattr(admin_api, "normalize_product_attributes", fake_normalize)
    response = await client.post(
        "/api/admin/assistant/catalog/normalize-attributes",
        headers={"X-Admin-Key": "test-secret"},
        json={"attributes": [{"name": "Питание", "value": "12V DC"}]},
    )

    assert response.status_code == 200
    assert response.json() == expected.model_dump()


async def test_admin_duplicate_candidates_returns_typed_review_list(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "test-secret")
    expected = DuplicateScanResponse(product_count=2, candidates=[])
    monkeypatch.setattr(admin_api, "find_duplicate_candidates", lambda request: expected)

    response = await client.post(
        "/api/admin/assistant/catalog/duplicate-candidates",
        headers={"X-Admin-Key": "test-secret"},
        json={"products": [
            {"name": "Датчик движения Livi"},
            {"name": "Датчик движения Livi беспроводной"},
        ]},
    )

    assert response.status_code == 200
    assert response.json() == expected.model_dump()


async def test_knowledge_upload_reports_full_process_index_as_retryable(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "test-secret")
    repository = KnowledgeRepository(max_documents=1, max_chunks=10, max_indexed_chars=100)
    await repository.add_document("existing", "Уже есть", "Исходные данные.")
    monkeypatch.setattr(admin_api, "knowledge_repository", repository)

    response = await client.post(
        "/api/admin/assistant/knowledge/documents",
        headers={"X-Admin-Key": "test-secret"},
        json={"document_id": "new", "title": "Новый", "text": "Новые данные."},
    )

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "60"


async def test_admin_document_upload_indexes_text_and_preserves_metadata(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "test-secret")
    document_id = f"test-upload-{uuid4().hex}"
    response = await client.post(
        "/api/admin/assistant/knowledge/documents/upload",
        headers={"X-Admin-Key": "test-secret"},
        data={
            "document_id": document_id,
            "title": "Инструкция по датчику",
            "product_sku": "SNS-100",
            "section": "Питание",
        },
        files={"file": ("sensor.txt", "Питание датчика: 12 В постоянного тока.", "text/plain")},
    )

    assert response.status_code == 201
    assert response.json()["chunks_added"] == 1
    search = await knowledge_repository.search("питание 12 В", "SNS-100")
    assert search.data["results"][0]["section"] == "Питание"


async def test_admin_document_upload_rejects_unsupported_extension(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "test-secret")
    response = await client.post(
        "/api/admin/assistant/knowledge/documents/upload",
        headers={"X-Admin-Key": "test-secret"},
        data={"document_id": "bad-upload", "title": "Bad file"},
        files={"file": ("catalog.xls", b"binary", "application/octet-stream")},
    )

    assert response.status_code == 415


async def test_checkout_validation_and_calculation_are_schema_checked(client):
    validation = await client.post(
        "/api/orders/validate-company-details",
        json={"inn": "7707083893", "ogrn": "1027700132195", "kpp": "770701001"},
    )
    calculation = await client.post(
        "/api/orders/calculate-total",
        json={
            "items": [{
                "sku": "SNS-1", "name": "Датчик", "quantity": 2, "unit_price": "100.00",
            }],
            "prior_purchase_total": "0",
            "discount_rules": [{
                "kind": "volume", "threshold": "150", "percent": "10", "label": "Объём",
            }],
            "stacking": "best",
        },
    )

    assert validation.status_code == 200
    assert validation.json()["is_valid"] is True
    assert validation.json()["existence_verified"] is False
    assert calculation.status_code == 200
    assert calculation.json()["subtotal"] == "200.00"
    assert calculation.json()["total"] == "180.00"


async def test_operator_request_is_queued_and_visible_only_in_admin_api(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "test-secret")
    create = await client.post(
        "/api/operator/requests",
        json={"reason": "Нужна консультация по выбору датчика"},
    )

    assert create.status_code == 202
    ticket = create.json()
    assert ticket["status"] == "queued_local"
    assert "оператор не подключён" in ticket["message"]

    listing = await client.get(
        "/api/admin/assistant/operator-requests",
        headers={"X-Admin-Key": "test-secret"},
    )

    assert listing.status_code == 200
    assert any(item["ticket_id"] == ticket["ticket_id"] for item in listing.json())


async def test_api_rate_limit_returns_retry_after(client, monkeypatch):
    monkeypatch.setattr(settings, "api_rate_limit_max_requests", 1)
    await rate_limiter.clear()
    body = {"inn": "7707083893"}
    first = await client.post("/api/orders/validate-company-details", json=body)
    second = await client.post("/api/orders/validate-company-details", json=body)
    await rate_limiter.clear()

    assert first.status_code == 200
    assert second.status_code == 429
    assert int(second.headers["Retry-After"]) >= 1


async def test_admin_content_controls_are_auth_protected_and_versioned(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "test-secret")
    store = ContentControlsStore()
    monkeypatch.setattr(admin_api, "content_controls", store)
    denied = await client.get("/api/admin/assistant/content")
    updated = await client.put(
        "/api/admin/assistant/content",
        headers={"X-Admin-Key": "test-secret"},
        json={
            "tone": "formal",
            "disabled_topics": [{"label": "Закрытая тема", "match_terms": ["закрытая тема"]}],
            "faqs": [{"question": "Есть паспорт?", "answer": "Он доступен в карточке товара."}],
        },
    )
    history = await client.get(
        "/api/admin/assistant/content/history",
        headers={"X-Admin-Key": "test-secret"},
    )

    assert denied.status_code == 401
    assert updated.status_code == 200
    assert updated.json()["revision"] == 2
    assert updated.json()["controls"]["tone"] == "formal"
    assert len(history.json()) == 2


async def test_customer_chat_stream_emits_tokens_and_done_event(client, monkeypatch):
    fake_graph = FakeStreamingGraph("Подбор готов")
    monkeypatch.setattr(customer_api, "_agent", lambda: fake_graph)

    response = await client.post(
        "/api/chat/stream",
        json={"message": "Подбери систему для дома"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        dict(line.split(": ", maxsplit=1) for line in block.splitlines())
        for block in response.text.strip().split("\n\n")
    ]
    token_text = "".join(
        json.loads(event["data"])["text"]
        for event in events
        if event["event"] == "token"
    )

    assert token_text == "Подбор готов"
    assert "event: done" in response.text
    assert '"conversation_id"' in response.text


async def test_customer_chat_continues_server_side_conversation(client, monkeypatch):
    fake_graph = FakeGraph("Первый ответ")
    monkeypatch.setattr(customer_api, "_agent", lambda: fake_graph)

    first = await client.post("/api/chat", json={"message": "Первый вопрос"})
    conversation_id = first.json()["conversation_id"]
    second = await client.post(
        "/api/chat",
        json={"message": "Уточнение", "conversation_id": conversation_id},
    )

    assert second.status_code == 200
    messages = fake_graph.received[0]["messages"]
    assert [message.content for message in messages] == [
        "Первый вопрос", "Первый ответ", "Уточнение",
    ]
    assert second.json()["conversation_id"] == conversation_id


async def test_customer_chat_rejects_unknown_conversation(client, monkeypatch):
    monkeypatch.setattr(customer_api, "_agent", lambda: FakeGraph("Ответ"))

    response = await client.post(
        "/api/chat", json={"message": "Продолжение", "conversation_id": "unknown-session"}
    )

    assert response.status_code == 404


async def test_customer_and_admin_conversations_are_isolated(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "test-secret")
    customer_graph = FakeGraph("Ответ клиента")
    admin_graph = FakeGraph("Ответ администратора")
    monkeypatch.setattr(customer_api, "_agent", lambda: customer_graph)
    monkeypatch.setattr(admin_api, "_agent", lambda: admin_graph)

    customer = await client.post("/api/chat", json={"message": "Вопрос клиента"})
    conversation_id = customer.json()["conversation_id"]
    response = await client.post(
        "/api/admin/assistant/chat",
        headers={"X-Admin-Key": "test-secret"},
        json={"message": "Вопрос администратора", "conversation_id": conversation_id},
    )

    assert response.status_code == 404
    assert [message.content for message in customer_graph.received[0]["messages"]] == ["Вопрос клиента"]
