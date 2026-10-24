"""LangGraph workflow for turning customer questions into FAQ candidates."""

import json
import logging
from time import perf_counter
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.agents.faq_schemas import (
    FAQAnalysisDraft,
    FAQAnalysisState,
    FAQClusterDraft,
    FAQTextDraft,
    JEvClassificationDraft,
)
from app.config import settings
from app.prompts.faq_analysis import (
    FAQ_ANALYSIS_SYSTEM_PROMPT,
)
from app.providers.chat_model import create_chat_model
from app.services.logging import log_event
from app.services.jev_classification import classify_faq_questions
from app.services.model_capacity import model_capacity

logger = logging.getLogger("app.agents.faq_analyzer")

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
            rows = json.loads(state["input_text"])
            classification = await classify_faq_questions(
                [row["question"] for row in rows]
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
        except Exception as exc:
            log_event(
                logger,
                logging.ERROR,
                "faq.classification_failed",
                llm_provider="typesafe",
                llm_model=settings.jev_llm_model,
                duration_ms=round((perf_counter() - started) * 1000, 2),
                error_type=type(exc).__name__,
            )
            raise

    async def write_text(state: FAQAnalysisState) -> dict[str, FAQTextDraft]:
        started = perf_counter()
        rows = json.loads(state["input_text"])
        assignments = state["classification"].assignments if state["classification"] else []
        categories: dict[int, str] = {}
        for assignment in assignments:
            if assignment.source_index not in categories:
                categories[assignment.source_index] = assignment.category

        for row in rows:
            row["category"] = categories.get(row["index"], "прочее")

        log_event(
            logger,
            logging.INFO,
            "faq.text_generation_started",
            llm_provider=settings.llm_provider,
            llm_model=settings.llm_model,
        )
        try:
            async with model_capacity.slot():
                text_analysis = await text_model.ainvoke([
                    SystemMessage(content=FAQ_ANALYSIS_SYSTEM_PROMPT),
                    HumanMessage(content=json.dumps(rows, ensure_ascii=False)),
                ], max_tokens=settings.llm_max_output_tokens)
            log_event(
                logger,
                logging.INFO,
                "faq.text_generation_completed",
                llm_provider=settings.llm_provider,
                llm_model=settings.llm_model,
                duration_ms=round((perf_counter() - started) * 1000, 2),
            )
            return {"text_analysis": text_analysis}
        except Exception as exc:
            log_event(
                logger,
                logging.ERROR,
                "faq.text_generation_failed",
                llm_provider=settings.llm_provider,
                llm_model=settings.llm_model,
                duration_ms=round((perf_counter() - started) * 1000, 2),
                error_type=type(exc).__name__,
            )
            raise

    def combine(state: FAQAnalysisState) -> dict[str, FAQAnalysisDraft]:
        classification = state["classification"]
        text_analysis = state["text_analysis"]
        if classification is None or text_analysis is None:
            raise RuntimeError("FAQ-анализатор не завершил оба этапа")

        rows = json.loads(state["input_text"])
        valid_indexes = {row["index"] for row in rows}
        category_by_index: dict[int, str] = {}
        for assignment in classification.assignments:
            if assignment.source_index in valid_indexes:
                category_by_index.setdefault(assignment.source_index, assignment.category)

        clusters: list[FAQClusterDraft] = []
        assigned_indexes: set[int] = set()
        for draft in text_analysis.clusters:
            indexes = [
                index for index in dict.fromkeys(draft.source_indexes)
                if index in valid_indexes and index not in assigned_indexes
            ]
            assigned_indexes.update(indexes)
            # Category always comes from JEv. Split a text model's mixed-category
            # cluster instead of letting its output overwrite the classification.
            indexes_by_category: dict[str, list[int]] = {}
            for index in indexes:
                category = category_by_index.get(index, "прочее")
                indexes_by_category.setdefault(category, []).append(index)
            for category, category_indexes in indexes_by_category.items():
                clusters.append(FAQClusterDraft(
                    source_indexes=category_indexes,
                    normalized_question=draft.normalized_question,
                    category=category,
                    faq_score=draft.faq_score,
                    rationale=draft.rationale,
                ))

        return {"analysis": FAQAnalysisDraft(summary=text_analysis.summary, clusters=clusters)}

    graph = StateGraph(FAQAnalysisState)
    graph.add_node("classify_questions", classify)
    graph.add_node("write_faq_text", write_text)
    graph.add_node("combine_results", combine)
    graph.add_edge(START, "classify_questions")
    graph.add_edge("classify_questions", "write_faq_text")
    graph.add_edge("write_faq_text", "combine_results")
    graph.add_edge("combine_results", END)
    return graph.compile()
