"""Pydantic DTOs for the standalone FAQ HTTP API."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QuestionSubmission(StrictModel):
    question: str = Field(min_length=3, max_length=1_000)
    conversation_id: str | None = Field(default=None, max_length=128)
    event_id: UUID | None = None


class CandidateResponse(StrictModel):
    id: int | None = None
    category: str
    question: str
    frequency: int
    score: int
    priority: str
    rationale: str
    source_questions: list[dict[str, object]] = Field(default_factory=list)
    published: bool = False


class PublicFAQItem(StrictModel):
    id: int
    category: str
    question: str


class AIAnalysisCluster(StrictModel):
    source_indexes: list[int] = Field(min_length=1)
    normalized_question: str = Field(min_length=1, max_length=1_000)
    category: Literal[
        "подбор и покупка",
        "характеристики и совместимость",
        "монтаж и настройка",
        "эксплуатация и неисправности",
        "гарантия и обслуживание",
        "доставка и оплата",
        "прочее",
    ]
    frequency: int = Field(ge=1)
    faq_score: int = Field(ge=0, le=100)
    priority: Literal["высокий", "средний", "низкий", "проверить"]
    rationale: str = Field(min_length=1, max_length=1_000)


class AIAnalysisResponse(StrictModel):
    summary: str = Field(min_length=1, max_length=1_500)
    question_count: int = Field(ge=1)
    clusters: list[AIAnalysisCluster] = Field(min_length=1)


def validate_ai_analysis_response(payload: object, *, expected_question_count: int) -> AIAnalysisResponse:
    """Validate both the HTTP contract and complete, non-duplicated source coverage."""
    try:
        analysis = AIAnalysisResponse.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("AI returned an invalid FAQ analysis response") from exc

    if analysis.question_count != expected_question_count:
        raise ValueError("AI FAQ response has an unexpected question count")

    indexes = [index for cluster in analysis.clusters for index in cluster.source_indexes]
    expected_indexes = set(range(1, expected_question_count + 1))
    if len(indexes) != expected_question_count or set(indexes) != expected_indexes:
        raise ValueError("AI FAQ response must assign every source exactly once")
    return analysis
