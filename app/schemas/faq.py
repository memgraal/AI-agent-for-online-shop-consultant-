"""FAQ analysis request and response contracts."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


FAQCategory = Literal[
    "подбор и покупка",
    "характеристики и совместимость",
    "монтаж и настройка",
    "эксплуатация и неисправности",
    "гарантия и обслуживание",
    "доставка и оплата",
    "прочее",
]
FAQPriority = Literal["высокий", "средний", "низкий", "проверить"]
FAQVariant = Annotated[str, Field(min_length=3, max_length=1_000)]
# Short grounded answers such as "Да" are valid context and must not make the
# whole FAQ batch fail validation.
FAQAnswerContext = Annotated[str, Field(min_length=1, max_length=4_000)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FAQQuestionInput(StrictModel):
    text: str = Field(min_length=3, max_length=1_000)
    frequency: int = Field(default=1, ge=1, le=100_000)
    variants: list[FAQVariant] = Field(default_factory=list, max_length=5)
    answer_context: list[FAQAnswerContext] = Field(default_factory=list, max_length=5)


class FAQAnalysisRequest(StrictModel):
    questions: list[FAQQuestionInput] = Field(min_length=1, max_length=100)


class FAQCluster(StrictModel):
    source_indexes: list[int]
    normalized_question: str
    category: FAQCategory
    frequency: int
    faq_score: int = Field(ge=0, le=100)
    priority: FAQPriority
    rationale: str
    answer: str = Field(default="", max_length=4_000)


class FAQAnalysisResponse(StrictModel):
    summary: str
    question_count: int
    clusters: list[FAQCluster]
