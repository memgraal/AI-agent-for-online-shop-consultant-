"""LLM structured output and LangGraph state for FAQ analysis."""

from typing import TypedDict

from pydantic import BaseModel, Field

from app.schemas.faq import FAQCategory


class FAQClusterDraft(BaseModel):
    source_indexes: list[int] = Field(min_length=1)
    normalized_question: str = Field(min_length=1, max_length=500)
    category: FAQCategory
    faq_score: int = Field(ge=0, le=100)
    rationale: str = Field(min_length=1, max_length=1000)
    answer: str = Field(default="", max_length=4000)


class FAQAnalysisDraft(BaseModel):
    summary: str = Field(min_length=1, max_length=1500)
    clusters: list[FAQClusterDraft]


class FAQCategoryAssignment(BaseModel):
    source_index: int = Field(ge=1)
    category: FAQCategory


class JEvClassificationDraft(BaseModel):
    assignments: list[FAQCategoryAssignment]


class FAQTextClusterDraft(BaseModel):
    source_indexes: list[int] = Field(min_length=1)
    normalized_question: str = Field(min_length=1, max_length=500)
    faq_score: int = Field(ge=0, le=100)
    rationale: str = Field(min_length=1, max_length=1000)
    answer: str = Field(default="", max_length=4000)


class FAQTextDraft(BaseModel):
    summary: str = Field(min_length=1, max_length=1500)
    clusters: list[FAQTextClusterDraft]


class FAQAnalysisState(TypedDict):
    input_text: str
    classification: JEvClassificationDraft | None
    text_analysis: FAQTextDraft | None
    analysis: FAQAnalysisDraft | None
