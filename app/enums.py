"""Closed domain values shared by API schemas and services."""

from enum import StrEnum


class FAQCategory(StrEnum):
    PRODUCT_SELECTION = "подбор и покупка"
    COMPATIBILITY = "характеристики и совместимость"
    INSTALLATION = "монтаж и настройка"
    OPERATION = "эксплуатация и неисправности"
    WARRANTY = "гарантия и обслуживание"
    DELIVERY = "доставка и оплата"
    OTHER = "прочее"


class FAQPriority(StrEnum):
    HIGH = "высокий"
    MEDIUM = "средний"
    LOW = "низкий"
    REVIEW = "проверить"


class ChatRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatDetail(StrEnum):
    QUICK = "quick"
    DETAILED = "detailed"


class Tone(StrEnum):
    FRIENDLY = "friendly"
    FORMAL = "formal"
    CONCISE = "concise"


class LLMProvider(StrEnum):
    OPENAI = "openai"
    YANDEX = "yandex"
    ROUTERAI = "routerai"
    OPENAI_COMPATIBLE = "openai-compatible"


class IdentifierKind(StrEnum):
    INN = "inn"
    OGRN = "ogrn"
    KPP = "kpp"


class DiscountKind(StrEnum):
    VOLUME = "volume"
    LOYALTY = "loyalty"


class DiscountStacking(StrEnum):
    BEST = "best"
    SEQUENTIAL = "sequential"


class OperatorStatus(StrEnum):
    QUEUED_LOCAL = "queued_local"


class Currency(StrEnum):
    RUB = "RUB"
