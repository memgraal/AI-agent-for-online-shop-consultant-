from langchain_core.tools import tool

from app.repositories.catalog import CatalogRepository
from app.repositories.knowledge import knowledge_repository
from app.repositories.orders import OrderRepository
from app.schemas.operator import OperatorRequest
from app.services.operator_handoff import operator_handoff_queue
from app.tools.observability import run_tool
from app.services.logging import log_event
import logging

catalog = CatalogRepository()
orders = OrderRepository()


@tool
async def search_catalog(query: str, object_type: str = "", area_m2: float | None = None,
                         requirements: str = "") -> str:
    """Подобрать товары каталога под задачу и параметры объекта."""
    return await run_tool("search_catalog", lambda: catalog.search(query, {
        "object_type": object_type, "area_m2": area_m2, "requirements": requirements,
    }))


@tool
async def compare_products(product_ids_or_skus: list[str], attributes: list[str]) -> str:
    """Сравнить указанные товары по характеристикам их карточек."""
    return await run_tool("compare_products", lambda: catalog.compare(product_ids_or_skus, attributes))


@tool
async def search_knowledge_base(query: str, product_or_sku: str = "") -> str:
    """Найти информацию в документах и базе знаний магазина."""
    return await run_tool("search_knowledge_base", lambda: knowledge_repository.search(query, product_or_sku))


@tool
async def validate_company_details(details: str) -> str:
    """Проверить формат реквизитов юрлица; не включайте лишние персональные данные."""
    return await run_tool("validate_company_details", lambda: orders.validate_company_details(details))


@tool
async def calculate_order_total(items: str, customer_type: str = "", purchase_total: float | None = None) -> str:
    """Рассчитать итоговую сумму корзины и применимые правила скидок."""
    return await run_tool("calculate_order_total", lambda: orders.calculate_total(items, customer_type, purchase_total))


@tool
async def call_operator(reason: str) -> str:
    """Создать локальную заявку оператору; внешний сервис пока не подключён."""
    ticket = await operator_handoff_queue.create(OperatorRequest(reason=reason))
    log_event(logging.getLogger("app.tools"), logging.INFO,
              "tool.completed", tool="call_operator", status=ticket.status)
    return ticket.model_dump_json()


@tool
async def create_installation_brief(
    object_type: str,
    area_m2: float | None = None,
    rooms_or_zones: list[str] | None = None,
    security_goals: str = "",
    communication_channels: list[str] | None = None,
    backup_autonomy_hours: float | None = None,
    constraints: str = "",
) -> str:
    """Собрать вводные в краткое ТЗ для консультации или передачи монтажнику."""
    from app.services.project_brief import render_installation_brief

    log_event(logging.getLogger("app.tools"), logging.INFO,
              "tool.started", tool="create_installation_brief")
    brief = render_installation_brief(
        object_type=object_type,
        area_m2=area_m2,
        rooms_or_zones=rooms_or_zones or [],
        security_goals=security_goals,
        communication_channels=communication_channels or [],
        backup_autonomy_hours=backup_autonomy_hours,
        constraints=constraints,
    )
    log_event(logging.getLogger("app.tools"), logging.INFO,
              "tool.completed", tool="create_installation_brief",
              answer_chars=len(brief))
    return brief


CUSTOMER_TOOLS = [search_catalog, compare_products, search_knowledge_base,
                  validate_company_details, calculate_order_total, call_operator,
                  create_installation_brief]
