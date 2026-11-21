import pytest
from decimal import Decimal
from typing import cast

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.repositories.analytics import AnalyticsRepository
from app.agents.tool_selection import select_customer_tools
from app.tools.customer import search_knowledge_base
from langchain_core.messages import HumanMessage
from app.repositories.catalog import CatalogRepository
from app.repositories.knowledge import KnowledgeCapacityExceeded, KnowledgeRepository
from app.repositories.orders import OrderRepository
from app.tools.customer import search_catalog


class _MappingRows:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _ReadSession:
    def __init__(self, rows):
        self.rows = rows
        self.parameters = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def execute(self, _statement, parameters):
        self.parameters = parameters
        return _MappingRows(self.rows)


class _SessionFactory:
    def __init__(self, session):
        self.session = session

    def __call__(self):
        return self.session


@pytest.mark.parametrize(
    ("repository", "method", "args"),
    [
        (CatalogRepository(), "search", ("alarm", {})),
        (CatalogRepository(), "compare", (["sku-1"], ["power"])),
        (OrderRepository(), "calculate_total", ("items", "person", 1000.0)),
        (AnalyticsRepository(), "summarize_questions", ("month",)),
    ],
)
async def test_async_repository_stubs_return_unconfigured(repository, method, args):
    result = await getattr(repository, method)(*args)

    assert result.configured is False
    assert "не подключён" in result.message


async def test_langchain_async_tool_awaits_async_repository_stub():
    result = await search_catalog.ainvoke({"query": "коттедж"})

    assert "не подключён" in result


def test_reviewed_faq_questions_select_knowledge_retrieval():
    selected = select_customer_tools({
        "messages": [HumanMessage(content="Какая гарантия и доставка?")],
        "detail": "quick",
        "llm_calls": 0,
    })

    assert search_knowledge_base in selected


async def test_catalog_search_returns_grounded_postgresql_product_facts():
    session = _ReadSession([
        {
            "id": 42,
            "name": "Ajax Hub 2",
            "sku": "AJ-HUB2",
            "manufacturer_part_number": "HUB2-W",
            "manufacturers": "Ajax",
            "categories": "Охранные панели",
            "short_description": "Централь охранной системы",
            "price": Decimal("24990.0000"),
            "old_price": Decimal("26990.0000"),
            "call_for_price": False,
            "stock_quantity": 7,
            "available_for_preorder": False,
            "specifications": "Каналы связи: Ethernet, LTE",
            "slug": "ajax-hub-2",
            "relevance": 2,
        }
    ])
    repository = CatalogRepository(
        cast(async_sessionmaker[AsyncSession], _SessionFactory(session))
    )

    result = await repository.search("Ajax Hub", {})

    assert result.configured is True
    assert result.data["source"] == "postgresql_store_catalog"
    assert result.data["results"][0]["sku"] == "AJ-HUB2"
    assert result.data["results"][0]["price"] == "24990.0000"
    assert result.data["results"][0]["in_stock"] is True
    assert session.parameters == {"query": "Ajax Hub", "limit": 8}


async def test_empty_knowledge_index_returns_no_results_without_claiming_source_is_missing():
    result = await KnowledgeRepository().search("manual", "sku-1")

    assert result.configured is True
    assert result.data["results"] == []


async def test_knowledge_search_returns_matching_passage_and_citation_metadata():
    repository = KnowledgeRepository()
    await repository.add_document(
        "manual-1", "Установка датчика", "Подключение питания 12 В постоянного тока.",
        product_sku="SKU-12", section="Питание", page=4,
        source_url="https://example.test/manual",
    )

    result = await repository.search("питания 12 В", "SKU-12")

    assert len(result.data["results"]) == 1
    citation = result.data["results"][0]
    assert citation["sku"] == "SKU-12"
    assert citation["page"] == 4
    assert citation["section"] == "Питание"
    assert citation["url"] == "https://example.test/manual"


async def test_knowledge_search_product_filter_matches_title_as_well_as_sku():
    repository = KnowledgeRepository()
    await repository.add_document(
        "manual-2", "Домофон VIZIT", "Устройство поддерживает вызов с панели.",
        product_sku="VZ-100",
    )

    result = await repository.search("вызов панели", "Домофон VIZIT")

    assert len(result.data["results"]) == 1
    assert result.data["results"][0]["title"] == "Домофон VIZIT"


async def test_knowledge_search_includes_only_retrieved_published_faq():
    session = _ReadSession([
        {
            "id": 3,
            "category": "гарантия и обслуживание",
            "normalized_question": "Какая гарантия на товар?",
            "answer": "Срок гарантии указан в карточке товара.",
            "relevance": 2,
        }
    ])
    repository = KnowledgeRepository(
        faq_session_factory=cast(
            async_sessionmaker[AsyncSession], _SessionFactory(session)
        )
    )

    result = await repository.search("гарантия на товар")

    assert result.data["results"][0]["source"] == "reviewed_published_faq"
    assert result.data["results"][0]["snippet"] == "Срок гарантии указан в карточке товара."


async def test_knowledge_index_rejects_capacity_overflow_without_evicting_existing_data():
    repository = KnowledgeRepository(max_documents=1, max_chunks=2, max_indexed_chars=100)
    await repository.add_document("doc-1", "Первый", "Сведения о питании 12 В.")

    with pytest.raises(KnowledgeCapacityExceeded):
        await repository.add_document("doc-2", "Второй", "Другой источник.")

    result = await repository.search("питании 12 В")
    assert len(result.data["results"]) == 1
    assert result.data["results"][0]["title"] == "Первый"
