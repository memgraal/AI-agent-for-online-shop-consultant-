"""Two-model product draft generation for the protected admin assistant."""

import json
import logging
from time import perf_counter

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field

from app.config import settings
from app.providers.chat_model import create_chat_model
from app.schemas.admin_product import (
    ProductClassificationDraft,
    ProductContentDraft,
    ProductSEODraft,
)
from app.services.jev_classification import classify_product as classify_product_with_jev
from app.services.agent_settings import agent_settings
from app.services.logging import log_event
from app.services.model_capacity import model_capacity

logger = logging.getLogger("app.services.admin_product")


class _ProductSource(BaseModel):
    name: str = Field(min_length=1, max_length=500)
    manufacturer: str = Field(min_length=1, max_length=300)
    attributes: str = Field(min_length=1, max_length=12000)


async def generate_product_content(
    name: str,
    manufacturer: str,
    attributes: str,
) -> ProductContentDraft:
    """Classify a product with JEv, then draft its copy with the chat model."""
    source = _ProductSource(
        name=name.strip(),
        manufacturer=manufacturer.strip(),
        attributes=attributes.strip(),
    )
    source_payload = source.model_dump()
    runtime = agent_settings.get().settings

    classification = await classify_product(
        source.name,
        source.manufacturer,
        source.attributes,
    )

    started = perf_counter()
    log_event(
        logger,
        logging.INFO,
        "admin.product_text_generation_started",
        llm_provider=runtime.llm_provider,
        llm_model=runtime.llm_model,
    )
    try:
        # Yandex's AliceAI model used here does not reliably return LangChain
        # function-call arguments for a schema. Its JSON-object response mode is
        # supported and lets Pydantic validate the actual generated object.
        output_method = "json_mode" if runtime.llm_provider == "yandex" else "function_calling"
        structured_seo_model = create_chat_model().with_structured_output(
            ProductSEODraft,
            method=output_method,
        )
        seo_messages = [
            SystemMessage(content=agent_settings.prompt("product_seo")),
            HumanMessage(content=json.dumps({
                **source_payload,
                "classification": classification.model_dump(),
            }, ensure_ascii=False)),
        ]
        try:
            async with model_capacity.slot():
                seo = await structured_seo_model.ainvoke(
                    seo_messages,
                    max_tokens=runtime.llm_max_output_tokens,
                )
        except OutputParserException:
            # AliceAI sometimes returns a syntactically valid but incomplete JSON
            # object. Retry once with an explicit field checklist; never accept a
            # partial draft or let the admin agent improvise one.
            log_event(
                logger,
                logging.WARNING,
                "admin.product_text_generation_retry",
                llm_provider=runtime.llm_provider,
                llm_model=runtime.llm_model,
            )
            retry_messages = [
                SystemMessage(content=agent_settings.prompt("product_seo")),
                HumanMessage(content=json.dumps({
                    **source_payload,
                    "classification": classification.model_dump(),
                }, ensure_ascii=False)),
                HumanMessage(content=(
                    "Предыдущий JSON был неполным. Повтори генерацию и верни "
                    "один JSON-объект со всеми пятью непустыми строковыми полями: "
                    "description, h1, meta_title, meta_description, category_text. "
                    "Не пропускай поля и не добавляй текст вне JSON."
                )),
            ]
            async with model_capacity.slot():
                seo = await structured_seo_model.ainvoke(
                    retry_messages,
                    max_tokens=runtime.llm_max_output_tokens,
                )
        log_event(
            logger,
            logging.INFO,
            "admin.product_text_generation_completed",
            llm_provider=runtime.llm_provider,
            llm_model=runtime.llm_model,
            duration_ms=round((perf_counter() - started) * 1000, 2),
        )
    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "admin.product_text_generation_failed",
            llm_provider=runtime.llm_provider,
            llm_model=runtime.llm_model,
            duration_ms=round((perf_counter() - started) * 1000, 2),
            error_type=type(exc).__name__,
        )
        raise

    return ProductContentDraft(
        classification=classification,
        seo=seo,
        related_product_criteria={
            "manufacturer": source.manufacturer,
            "product_type": classification.product_type,
            "purpose": classification.purpose,
        },
    )


async def classify_product(
    name: str,
    manufacturer: str,
    attributes: str,
) -> ProductClassificationDraft:
    """Return a validated JEv classification for one product."""
    source = _ProductSource(
        name=name.strip(),
        manufacturer=manufacturer.strip(),
        attributes=attributes.strip(),
    )
    started = perf_counter()
    log_event(
        logger,
        logging.INFO,
        "admin.product_classification_started",
        llm_provider="typesafe",
        llm_model=settings.jev_llm_model,
    )
    try:
        classification = await classify_product_with_jev(
            name=source.name,
            manufacturer=source.manufacturer,
            attributes=source.attributes,
        )
        log_event(
            logger,
            logging.INFO,
            "admin.product_classification_completed",
            llm_provider="typesafe",
            llm_model=settings.jev_llm_model,
            duration_ms=round((perf_counter() - started) * 1000, 2),
        )
    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "admin.product_classification_failed",
            llm_provider="typesafe",
            llm_model=settings.jev_llm_model,
            duration_ms=round((perf_counter() - started) * 1000, 2),
            error_type=type(exc).__name__,
        )
        raise
    return classification
