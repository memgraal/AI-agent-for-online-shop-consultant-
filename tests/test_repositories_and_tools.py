import pytest

from app.repositories.analytics import AnalyticsRepository
from app.repositories.catalog import CatalogRepository
from app.repositories.knowledge import KnowledgeCapacityExceeded, KnowledgeRepository
from app.repositories.orders import OrderRepository
from app.tools.customer import search_catalog


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


async def test_knowledge_index_rejects_capacity_overflow_without_evicting_existing_data():
    repository = KnowledgeRepository(max_documents=1, max_chunks=2, max_indexed_chars=100)
    await repository.add_document("doc-1", "Первый", "Сведения о питании 12 В.")

    with pytest.raises(KnowledgeCapacityExceeded):
        await repository.add_document("doc-2", "Второй", "Другой источник.")

    result = await repository.search("питании 12 В")
    assert len(result.data["results"]) == 1
    assert result.data["results"][0]["title"] == "Первый"
