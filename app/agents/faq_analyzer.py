"""LangGraph workflow for turning customer questions into FAQ candidates."""

import json
import logging
from time import perf_counter
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.agents.faq_schemas import (
    FAQAnalysisDraft,
    FAQAnalysisState,
    FAQClusterDraft,
    FAQCategoryAssignment,
    FAQTextClusterDraft,
    FAQTextDraft,
    JEvClassificationDraft,
)
from app.config import settings
from app.providers.chat_model import create_chat_model
from app.schemas.faq import FAQCategory
from app.services.agent_settings import agent_settings
from app.services.logging import log_event
from app.services.jev_classification import classify_faq_questions
from app.services.model_capacity import ModelOverloadedError, model_capacity

logger = logging.getLogger("app.agents.faq_analyzer")


def _row_index(row: dict[str, object]) -> int:
    value = row.get("index")
    if isinstance(value, bool):
        raise ValueError("FAQ source index must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    raise ValueError("FAQ source index must be an integer")


def _row_question(row: dict[str, object]) -> str:
    value = row.get("question", row.get("text", ""))
    return value if isinstance(value, str) else str(value)


def _row_answers(row: dict[str, object]) -> list[str]:
    value = row.get("answer_context", [])
    if not isinstance(value, list):
        return []
    return [str(answer) for answer in value]


def _decode_rows(input_text: str) -> list[dict[str, object]]:
    value: Any = json.loads(input_text)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("FAQ analyzer input must be a list of objects")
    return value


def _fallback_text_analysis(rows: list[dict[str, object]]) -> FAQTextDraft:
    """Keep refused/failed model output reviewable without inventing answers."""
    clusters = [
        FAQTextClusterDraft(
            source_indexes=[_row_index(row)],
            normalized_question=_row_question(row),
            faq_score=0,
            rationale="Основная модель не сформировала ответ; требуется ручная проверка.",
            answer=next(
                (
                    str(answer).strip()
                    for answer in _row_answers(row)
                    if str(answer).strip()
                ),
                "",
            ),
        )
        for row in rows
    ]
    return FAQTextDraft(
        summary="Модель не сформировала полный анализ; вопросы оставлены на ручную проверку.",
        clusters=clusters,
    )


def build_faq_analyzer():
    """Build a two-stage FAQ flow: JEv categories, chat model writes text."""
    text_model = create_chat_model().with_structured_output(
        FAQTextDraft,
        method="function_calling",
    )

    async def classify(state: FAQAnalysisState) -> dict[str, JEvClassificationDraft]:
        started = perf_counter()
        log_event(
            logger,
            logging.INFO,
            "faq.classification_started",
            llm_provider="typesafe",
            llm_model=settings.jev_llm_model,
        )
        try:
            rows = _decode_rows(state["input_text"])
            classification = await classify_faq_questions(
                [_row_question(row) for row in rows]
            )
            log_event(
                logger,
                logging.INFO,
                "faq.classification_completed",
                llm_provider="typesafe",
                llm_model=settings.jev_llm_model,
                duration_ms=round((perf_counter() - started) * 1000, 2),
            )
            return {"classification": classification}
        except ModelOverloadedError:
            raise
        except Exception as exc:
            log_event(
                logger,
                logging.WARNING,
                "faq.classification_fallback",
                llm_provider="typesafe",
                llm_model=settings.jev_llm_model,
                duration_ms=round((perf_counter() - started) * 1000, 2),
                error_type=type(exc).__name__,
            )
            rows = _decode_rows(state["input_text"])
            return {
                "classification": JEvClassificationDraft(
                    assignments=[
                        FAQCategoryAssignment(
                            source_index=_row_index(row), category="прочее"
                        )
                        for row in rows
                    ]
                )
            }

    async def write_text(state: FAQAnalysisState) -> dict[str, FAQTextDraft]:
        started = perf_counter()
        rows = _decode_rows(state["input_text"])
        assignments = (
            state["classification"].assignments if state["classification"] else []
        )
        categories: dict[int, FAQCategory] = {}
        for assignment in assignments:
            if assignment.source_index not in categories:
                categories[assignment.source_index] = assignment.category

        for row in rows:
            row["category"] = categories.get(_row_index(row), "прочее")

        runtime = agent_settings.get().settings
        log_event(
            logger,
            logging.INFO,
            "faq.text_generation_started",
            llm_provider=runtime.llm_provider,
            llm_model=runtime.llm_model,
        )
        try:
            async with model_capacity.slot():
                text_analysis = await text_model.ainvoke(
                    [
                        SystemMessage(content=agent_settings.prompt("faq_analysis")),
                        HumanMessage(content=json.dumps(rows, ensure_ascii=False)),
                    ],
                    max_tokens=runtime.llm_max_output_tokens,
                )
            log_event(
                logger,
                logging.INFO,
                "faq.text_generation_completed",
                llm_provider=runtime.llm_provider,
                llm_model=runtime.llm_model,
                duration_ms=round((perf_counter() - started) * 1000, 2),
            )
            if not isinstance(text_analysis, FAQTextDraft):
                text_analysis = FAQTextDraft.model_validate(text_analysis)
            return {"text_analysis": text_analysis}
        except ModelOverloadedError:
            raise
        except Exception as exc:
            log_event(
                logger,
                logging.WARNING,
                "faq.text_generation_fallback",
                llm_provider=runtime.llm_provider,
                llm_model=runtime.llm_model,
                duration_ms=round((perf_counter() - started) * 1000, 2),
                error_type=type(exc).__name__,
            )
            return {"text_analysis": _fallback_text_analysis(rows)}

    def combine(state: FAQAnalysisState) -> dict[str, FAQAnalysisDraft]:
        classification = state["classification"]
        text_analysis = state["text_analysis"]
        if classification is None or text_analysis is None:
            raise RuntimeError("FAQ-анализатор не завершил оба этапа")

        rows = _decode_rows(state["input_text"])
        valid_indexes = {_row_index(row) for row in rows}
        category_by_index: dict[int, FAQCategory] = {}
        for assignment in classification.assignments:
            if assignment.source_index in valid_indexes:
                category_by_index.setdefault(
                    assignment.source_index, assignment.category
                )

        clusters: list[FAQClusterDraft] = []
        assigned_indexes: set[int] = set()
        for draft in text_analysis.clusters:
            indexes = [
                index
                for index in dict.fromkeys(draft.source_indexes)
                if index in valid_indexes and index not in assigned_indexes
            ]
            assigned_indexes.update(indexes)
            # Category always comes from JEv. Split a text model's mixed-category
            # cluster instead of letting its output overwrite the classification.
            indexes_by_category: dict[FAQCategory, list[int]] = {}
            for index in indexes:
                category = category_by_index.get(index, "прочее")
                indexes_by_category.setdefault(category, []).append(index)
            for category, category_indexes in indexes_by_category.items():
                clusters.append(
                    FAQClusterDraft(
                        source_indexes=category_indexes,
                        normalized_question=draft.normalized_question,
                        category=category,
                        faq_score=draft.faq_score,
                        rationale=draft.rationale,
                        answer=draft.answer,
                    )
                )

        return {
            "analysis": FAQAnalysisDraft(
                summary=text_analysis.summary, clusters=clusters
            )
        }

    graph = StateGraph(FAQAnalysisState)
    graph.add_node("classify_questions", classify)
    graph.add_node("write_faq_text", write_text)
    graph.add_node("combine_results", combine)
    graph.add_edge(START, "classify_questions")
    graph.add_edge("classify_questions", "write_faq_text")
    graph.add_edge("write_faq_text", "combine_results")
    graph.add_edge("combine_results", END)
    return graph.compile()
