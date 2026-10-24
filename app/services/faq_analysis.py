"""FAQ analysis use case: invoke the LangGraph and validate its grouping."""

import json
from functools import lru_cache
from typing import Literal

from app.agents.faq_analyzer import build_faq_analyzer
from app.agents.faq_schemas import FAQAnalysisDraft
from app.config import settings
from app.schemas.faq import FAQAnalysisResponse, FAQCluster, FAQQuestionInput
from app.services.privacy import redact_personal_data


@lru_cache(maxsize=1)
def _analyzer():
    return build_faq_analyzer()


def _priority(score: int) -> Literal["высокий", "средний", "низкий"]:
    if score >= 70:
        return "высокий"
    if score >= 40:
        return "средний"
    return "низкий"


def _build_response(
    analysis: FAQAnalysisDraft,
    questions: list[FAQQuestionInput],
) -> FAQAnalysisResponse:
    clusters: list[FAQCluster] = []
    assigned_indexes: set[int] = set()

    for draft in analysis.clusters:
        valid_indexes = [
            index
            for index in dict.fromkeys(draft.source_indexes)
            if 1 <= index <= len(questions) and index not in assigned_indexes
        ]
        if not valid_indexes:
            continue

        assigned_indexes.update(valid_indexes)
        clusters.append(
            FAQCluster(
                source_indexes=valid_indexes,
                normalized_question=draft.normalized_question,
                category=draft.category,
                frequency=sum(questions[index - 1].frequency for index in valid_indexes),
                faq_score=draft.faq_score,
                priority=_priority(draft.faq_score),
                rationale=draft.rationale,
            )
        )

    # Keep every source question visible if the model omits it or returns bad indexes.
    for index, question in enumerate(questions, start=1):
        if index in assigned_indexes:
            continue
        clusters.append(
            FAQCluster(
                source_indexes=[index],
                normalized_question=redact_personal_data(question.text),
                category="прочее",
                frequency=question.frequency,
                faq_score=0,
                priority="проверить",
                rationale="Модель не распределила вопрос; проверьте вручную.",
            )
        )

    return FAQAnalysisResponse(
        summary=analysis.summary,
        question_count=len(questions),
        clusters=clusters,
    )


async def analyze_faq_questions(
    questions: list[FAQQuestionInput],
) -> FAQAnalysisResponse:
    """Group, categorize, and rank supplied questions for human FAQ review."""
    safe_questions = [
        {
            "index": index,
            "frequency": question.frequency,
            "question": redact_personal_data(question.text),
            "variants": [redact_personal_data(variant) for variant in question.variants],
        }
        for index, question in enumerate(questions, start=1)
    ]
    input_text = json.dumps(safe_questions, ensure_ascii=False)
    result = await _analyzer().ainvoke(
        {
            "input_text": input_text,
            "classification": None,
            "text_analysis": None,
            "analysis": None,
        },
        config={"recursion_limit": settings.agent_recursion_limit},
    )
    analysis = result.get("analysis")
    if analysis is None:
        raise RuntimeError("FAQ-анализатор вернул пустой результат")
    return _build_response(analysis, questions)
