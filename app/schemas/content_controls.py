"""Validated administrative controls for runtime assistant content."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DisabledTopic(StrictModel):
    label: str = Field(min_length=2, max_length=100)
    match_terms: list[str] = Field(min_length=1, max_length=20)

    @field_validator("match_terms")
    @classmethod
    def validate_match_terms(cls, terms: list[str]) -> list[str]:
        normalized = [term.strip() for term in terms]
        if any(len(term) < 2 for term in normalized):
            raise ValueError("Каждый термин отключаемой темы должен содержать минимум 2 символа.")
        return normalized


class ManagedFAQ(StrictModel):
    question: str = Field(min_length=3, max_length=500)
    answer: str = Field(min_length=3, max_length=3000)
    enabled: bool = True


class ContentControlsInput(StrictModel):
    tone: Literal["friendly", "formal", "concise"] = "friendly"
    disabled_topics: list[DisabledTopic] = Field(default_factory=list, max_length=30)
    faqs: list[ManagedFAQ] = Field(default_factory=list, max_length=200)
    disabled_topic_message: str = Field(
        default="По этой теме консультант сейчас не отвечает.",
        min_length=3,
        max_length=300,
    )


class ContentControlsSnapshot(StrictModel):
    revision: int = Field(ge=1)
    updated_at: datetime
    controls: ContentControlsInput
