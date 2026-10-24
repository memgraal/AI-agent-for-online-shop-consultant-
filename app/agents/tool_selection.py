"""Lightweight, deterministic selection of customer tools for each model call."""

from langchain_core.messages import HumanMessage
import re

from app.agents.state import AssistantState
from app.tools.customer import (
    CUSTOMER_TOOLS,
    calculate_order_total,
    call_operator,
    compare_products,
    create_installation_brief,
    search_catalog,
    search_knowledge_base,
    validate_company_details,
)


_TOOL_GROUPS = {
    "catalog": (search_catalog, search_knowledge_base),
    "comparison": (compare_products, search_knowledge_base),
    "knowledge": (search_knowledge_base,),
    "checkout": (calculate_order_total, validate_company_details),
    "company_details": (validate_company_details,),
    "installation": (create_installation_brief, search_catalog),
    "operator": (call_operator,),
}

_KEYWORDS = {
    "catalog": (
        "подобрать", "подбор", "найти товар", "найди товар", "посоветуй",
        "каталог", "купить", "налич", "цена", "стоимость", "артикул",
        "товар", "оборудован", "камера", "видеонаблюд", "датчик", "сигнализац",
        "ajax", "jablotron", "hub ", "ja-", "охран",
    ),
    "comparison": ("сравни", "сравнить", "сравнение", "отлич", "что лучше", "между"),
    "knowledge": (
        "инструкц", "паспорт", "сертификат", "совместим", "характеристик",
        "мегапиксел", "разрешени", "подключ", "настро", "монтаж", "как работает",
    ),
    "checkout": ("корзин", "скидк", "итогов", "посчитать", "рассчита", "сумм заказа"),
    "company_details": ("реквизит", "инн", "огрн", "юридическ лиц"),
    "installation": (
        "тз", "техническое задание", "монтажник", "объект", "площад", "помещен",
        "зон", "автономност", "резервное питание",
    ),
    "operator": ("оператор", "специалист", "сотрудник", "поддержк", "человек"),
}


def select_customer_tools(state: AssistantState) -> list:
    """Return only tools relevant to the latest turn; no extra model call is made."""
    user_text = _latest_user_text(state)
    if not user_text:
        return []

    groups = {
        group
        for group, keywords in _KEYWORDS.items()
        if any(keyword in user_text for keyword in keywords)
    }

    # Follow-up answers such as “Hub 2 Plus и JA-103K” inherit the intent of
    # the immediately preceding question, without carrying stale intents forward.
    if not groups and len(user_text) <= 80 and not re.fullmatch(r"[\d\s+*/().=-]+", user_text):
        previous_user_text = _previous_user_text(state)
        groups = {
            group
            for group, keywords in _KEYWORDS.items()
            if any(keyword in previous_user_text for keyword in keywords)
        }

    selected = []
    for group in _TOOL_GROUPS:
        if group in groups:
            selected.extend(_TOOL_GROUPS[group])

    # Preserve declaration order and remove duplicates for stable tool schemas.
    selected_names = {tool.name for tool in selected}
    return [tool for tool in CUSTOMER_TOOLS if tool.name in selected_names]


def _latest_user_text(state: AssistantState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return _message_text(message.content).lower()
    return ""


def _previous_user_text(state: AssistantState) -> str:
    found_latest = False
    previous_turns = []
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            if found_latest:
                previous_turns.append(_message_text(message.content).lower())
                if len(previous_turns) == 2:
                    break
                continue
            found_latest = True
    return " ".join(previous_turns)


def _message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    return ""
