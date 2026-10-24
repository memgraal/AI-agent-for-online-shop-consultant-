import json

from langchain_core.tools import tool

from app.repositories.analytics import AnalyticsRepository
from app.schemas.admin_catalog import (
    AttributeNormalizationRequest,
    DuplicateScanRequest,
)
from app.services.admin_catalog import find_duplicate_candidates, normalize_product_attributes
from app.services.admin_product import generate_product_content
from app.tools.observability import run_tool

analytics = AnalyticsRepository()


@tool
async def generate_product_seo(name: str, manufacturer: str, attributes: str) -> str:
    """Классифицировать товар через JEv и подготовить SEO-черновик основной моделью."""
    draft = await generate_product_content(name, manufacturer, attributes)
    return json.dumps(draft.model_dump(), ensure_ascii=False)


@tool
async def normalize_attributes(values: str) -> str:
    """Предложить нормализацию атрибутов импорта без изменения исходных данных."""
    request = AttributeNormalizationRequest.model_validate({"attributes": json.loads(values)})
    result = await normalize_product_attributes(request)
    return result.model_dump_json()


@tool
async def find_product_duplicates(products: str) -> str:
    """Найти потенциальные дубли карточек для проверки администратором."""
    request = DuplicateScanRequest.model_validate({"products": json.loads(products)})
    return find_duplicate_candidates(request).model_dump_json()


@tool
async def summarize_customer_questions(period: str = "") -> str:
    """Подготовить сводку частых вопросов по обезличенным обращениям за период."""
    return await run_tool("summarize_customer_questions", lambda: analytics.summarize_questions(period))


ADMIN_TOOLS = [generate_product_seo, normalize_attributes,
               find_product_duplicates, summarize_customer_questions]
