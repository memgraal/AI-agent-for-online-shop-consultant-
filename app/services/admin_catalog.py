"""Database-free catalog normalization and duplicate-candidate services."""

from difflib import SequenceMatcher
import json
import re
import unicodedata

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.providers.chat_model import create_chat_model
from app.schemas.admin_catalog import (
    AttributeNormalizationRequest,
    AttributeNormalizationResponse,
    DuplicateCandidate,
    DuplicateScanRequest,
    DuplicateScanResponse,
    NormalizedAttribute,
)
from app.services.model_capacity import model_capacity

_NORMALIZATION_PROMPT = """Ты — помощник администратора по нормализации товарных атрибутов.
Предложи канонические название и значение каждого атрибута. Сохраняй смысл и исходные
числа, единицы и обозначения; не добавляй отсутствующие характеристики. Верни ровно
по одному результату в исходном порядке. Уверенность укажи от 0 до 1. Все значения —
предложения для проверки администратором, исходные данные не изменяй.
"""


async def normalize_product_attributes(
    request: AttributeNormalizationRequest,
) -> AttributeNormalizationResponse:
    """Use the main model to suggest typed, reviewable attribute normalization."""
    payload = [item.model_dump() for item in request.attributes]
    model = create_chat_model().with_structured_output(
        AttributeNormalizationResponse,
        method="json_mode" if settings.llm_provider == "yandex" else "function_calling",
    )
    messages = [
        SystemMessage(content=_NORMALIZATION_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]
    try:
        async with model_capacity.slot():
            result = await model.ainvoke(messages, max_tokens=settings.llm_max_output_tokens)
    except OutputParserException:
        retry_messages = [
            *messages,
            HumanMessage(content=(
                "Предыдущий ответ не прошёл проверку. Верни один JSON-объект "
                "со списком attributes той же длины и в том же порядке. Для каждого "
                "элемента обязательны source_name, source_value, normalized_name, "
                "normalized_value, explanation и confidence."
            )),
        ]
        async with model_capacity.slot():
            result = await model.ainvoke(
                retry_messages,
                max_tokens=settings.llm_max_output_tokens,
            )

    if len(result.attributes) != len(request.attributes):
        raise ValueError("Модель вернула другое количество нормализованных атрибутов.")

    # The source values are authoritative. Never let a model alter the values shown
    # as the input side of the proposed mapping.
    normalized = [
        item.model_copy(update={
            "source_name": source.name,
            "source_value": source.value,
        })
        for source, item in zip(request.attributes, result.attributes, strict=True)
    ]
    return AttributeNormalizationResponse(attributes=normalized)


def find_duplicate_candidates(
    request: DuplicateScanRequest,
) -> DuplicateScanResponse:
    """Rank likely duplicates in a supplied batch; never merge or mutate records."""
    candidates: list[DuplicateCandidate] = []
    for left_index, left in enumerate(request.products):
        for right_index in range(left_index + 1, len(request.products)):
            right = request.products[right_index]
            left_sku = _canonical(left.sku)
            right_sku = _canonical(right.sku)
            name_similarity = _similarity(left.name, right.name)
            manufacturer_similarity = _similarity(left.manufacturer, right.manufacturer)
            attribute_similarity = _attribute_similarity(left.attributes, right.attributes)

            if left_sku and right_sku and left_sku == right_sku:
                score = 100
                matched_fields = ["sku"]
                rationale = "Совпадает артикул."
            else:
                score = round(
                    70 * name_similarity
                    + 20 * manufacturer_similarity
                    + 10 * attribute_similarity
                )
                matched_fields = []
                if name_similarity >= 0.82:
                    matched_fields.append("name")
                if manufacturer_similarity >= 0.8 and left.manufacturer and right.manufacturer:
                    matched_fields.append("manufacturer")
                if attribute_similarity >= 0.5:
                    matched_fields.append("attributes")
                # Require strong name evidence so identical manufacturers alone do
                # not produce a noisy list of duplicate candidates.
                if name_similarity < 0.82 or score < 78:
                    continue
                rationale = (
                    f"Похожее название ({name_similarity:.0%}); "
                    f"совпадение производителя {manufacturer_similarity:.0%}, "
                    f"атрибутов {attribute_similarity:.0%}."
                )

            candidates.append(DuplicateCandidate(
                left_index=left_index,
                right_index=right_index,
                left_sku=left.sku,
                right_sku=right.sku,
                left_name=left.name,
                right_name=right.name,
                confidence=score,
                matched_fields=matched_fields,
                rationale=rationale,
            ))

    candidates.sort(key=lambda candidate: (-candidate.confidence, candidate.left_index, candidate.right_index))
    return DuplicateScanResponse(
        product_count=len(request.products),
        candidates=candidates,
    )


def _canonical(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
    return re.sub(r"[^\w]+", "", normalized, flags=re.UNICODE)


def _similarity(left: str, right: str) -> float:
    left_value = _canonical(left)
    right_value = _canonical(right)
    if not left_value or not right_value:
        return 0.0
    return SequenceMatcher(None, left_value, right_value, autojunk=False).ratio()


def _attribute_similarity(left: dict[str, str], right: dict[str, str]) -> float:
    if not left or not right:
        return 0.0
    left_values = {_canonical(key): _canonical(value) for key, value in left.items()}
    right_values = {_canonical(key): _canonical(value) for key, value in right.items()}
    common_keys = left_values.keys() & right_values.keys()
    if not common_keys:
        return 0.0
    return sum(left_values[key] == right_values[key] for key in common_keys) / len(common_keys)
