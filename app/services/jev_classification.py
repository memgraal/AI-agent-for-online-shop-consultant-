"""Typed classification workflows backed by TypeSafe JEv."""

import json
import re
from collections.abc import Sequence
from typing import cast

from typesafe_sdk import Choice, Noul

from app.agents.faq_schemas import FAQCategoryAssignment, JEvClassificationDraft
from app.constants import (
    FAQ_CATEGORIES,
    PRODUCT_CATEGORIES,
    PRODUCT_PURPOSES,
    PRODUCT_TYPES,
)
from app.providers.jev_client import create_jev_client
from app.schemas.admin_product import ProductClassificationDraft
from app.schemas.faq import FAQCategory
from app.services.model_capacity import model_capacity

def _as_faq_category(value: object) -> FAQCategory:
    """Validate and narrow a provider string to the configured FAQ category."""
    if isinstance(value, str) and value in FAQ_CATEGORIES:
        return cast(FAQCategory, value)
    raise ValueError(f"JEv returned an unsupported FAQ category: {value!r}")


async def classify_faq_questions(questions: Sequence[str]) -> JEvClassificationDraft:
    """Assign one configured FAQ category to every question using JEv Choice."""
    if not questions:
        return JEvClassificationDraft(assignments=[])

    choice_questions = {
        f"question_{index}": Choice(
            instructions="Выбери ровно одну наиболее подходящую категорию вопроса.",
            criteria={category: category for category in FAQ_CATEGORIES},
        )
        for index, _ in enumerate(questions, start=1)
    }
    state = json.dumps(
        [
            {"source_index": index, "question": question}
            for index, question in enumerate(questions, start=1)
        ],
        ensure_ascii=False,
    )

    async with create_jev_client() as client:
        async with model_capacity.slot():
            response = await client.system_one(state=state, questions=choice_questions)

    assignments = [
        FAQCategoryAssignment(
            source_index=index,
            category=_as_faq_category(response.choices[f"question_{index}"].choice),
        )
        for index in range(1, len(questions) + 1)
    ]
    return JEvClassificationDraft(assignments=assignments)


def _tag_candidates(name: str, manufacturer: str, attributes: str) -> list[str]:
    """Extract bounded, source-grounded tag candidates for JEv to select."""
    candidates: list[str] = []
    for value in (name, manufacturer, attributes):
        for part in re.split(r"[,;\n|]+", value):
            tag = " ".join(part.split()).strip(" .:-")
            if len(tag) < 2 or tag.casefold() in {
                item.casefold() for item in candidates
            }:
                continue
            candidates.append(tag[:120])
            if len(candidates) >= 30:
                return candidates
    return candidates


async def classify_product(
    *,
    name: str,
    manufacturer: str,
    attributes: str,
) -> ProductClassificationDraft:
    """Classify a product with JEv's finite-choice and yes/no primitives."""
    candidates = _tag_candidates(name, manufacturer, attributes)
    questions = {
        "category": Choice(
            instructions="Выбери одну категорию товара из фиксированного списка.",
            criteria=PRODUCT_CATEGORIES,
        ),
        "product_type": Choice(
            instructions="Выбери наиболее точный тип товара из фиксированного списка.",
            criteria=PRODUCT_TYPES,
        ),
        "purpose": Choice(
            instructions="Выбери основное назначение товара из фиксированного списка.",
            criteria=PRODUCT_PURPOSES,
        ),
        **{
            f"tag_{index}": Noul(
                instructions=(
                    f"Является ли точная формулировка «{tag}» уместным поисковым тегом "
                    "этого товара, исходя только из переданных данных?"
                )
            )
            for index, tag in enumerate(candidates, start=1)
        },
    }
    state = json.dumps(
        {"name": name, "manufacturer": manufacturer, "attributes": attributes},
        ensure_ascii=False,
    )

    async with create_jev_client() as client:
        async with model_capacity.slot():
            response = await client.system_one(state=state, questions=questions)

    return ProductClassificationDraft(
        category=response.choices["category"].choice,
        product_type=response.choices["product_type"].choice,
        purpose=response.choices["purpose"].choice,
        tags=[
            tag
            for index, tag in enumerate(candidates, start=1)
            if response.nouls[f"tag_{index}"].noul >= 0.5
        ],
    )
