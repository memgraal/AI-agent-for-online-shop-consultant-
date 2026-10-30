"""FAQ analysis use case: invoke the LangGraph and validate its grouping."""

import json
from functools import lru_cache
from typing import Literal

from app.agents.faq_analyzer import build_faq_analyzer
from app.agents.faq_schemas import FAQAnalysisDraft
from app.schemas.faq import FAQAnalysisResponse, FAQCluster, FAQQuestionInput
from app.services.agent_settings import agent_settings
from app.services.privacy import redact_personal_data


_REFUSAL_MARKERS = (
    "не могу",
    "не знаю",
    "недостаточно информации",
    "нет информации",
    "не удалось определить",
    "не располагаю информацией",
    "не располагаю данными",
    "не могу помочь",
    "как ии",
    "i don't know",
    "i cannot",
    "i can't",
    "i'm unable",
    "as an ai",
    "insufficient information",
    "cannot determine",
)


def _is_refusal(answer: str) -> bool:
    normalized = " ".join(answer.casefold().split())
    return not normalized or any(marker in normalized for marker in _REFUSAL_MARKERS)


def _answer_from_context(
    questions: list[FAQQuestionInput],
    indexes: list[int],
) -> str:
    """Return the first grounded answer when the model refused or omitted one."""
    for index in indexes:
        for answer in questions[index - 1].answer_context:
            cleaned = answer.strip()
            if cleaned and not _is_refusal(cleaned):
                return cleaned
    return ""


def _resolve_answer(
    answer: str,
    questions: list[FAQQuestionInput],
    indexes: list[int],
) -> str:
    if not _is_refusal(answer):
        return answer.strip()
    return _answer_from_context(questions, indexes)


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
        answer = _resolve_answer(draft.answer, questions, valid_indexes)
        priority = _priority(draft.faq_score) if answer else "проверить"
        rationale = draft.rationale
        if not answer:
            rationale = (
                f"{rationale} Проверенный ответ отсутствует; требуется ручная проверка."
            )
        clusters.append(
            FAQCluster(
                source_indexes=valid_indexes,
                normalized_question=draft.normalized_question,
                category=draft.category,
                frequency=sum(
                    questions[index - 1].frequency for index in valid_indexes
                ),
                faq_score=draft.faq_score,
                priority=priority,
                rationale=rationale,
                answer=answer,
            )
        )

    # Keep every source question visible if the model omits it or returns bad indexes.
    for index, question in enumerate(questions, start=1):
        if index in assigned_indexes:
            continue
        answer = _answer_from_context(questions, [index])
        clusters.append(
            FAQCluster(
                source_indexes=[index],
                normalized_question=redact_personal_data(question.text),
                category="прочее",
                frequency=question.frequency,
                faq_score=0,
                priority="проверить",
                rationale="Модель не распределила вопрос; проверьте вручную.",
                answer=answer,
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
            "variants": [
                redact_personal_data(variant) for variant in question.variants
            ],
            "answer_context": [
                redact_personal_data(answer) for answer in question.answer_context
            ],
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
        config={"recursion_limit": agent_settings.get().settings.agent_recursion_limit},
    )
    analysis = result.get("analysis")
    if analysis is None:
        raise RuntimeError("FAQ-анализатор вернул пустой результат")
    return _build_response(analysis, questions)
