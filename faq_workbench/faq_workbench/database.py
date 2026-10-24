"""Async PostgreSQL schema and session lifecycle for the FAQ service."""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class FAQQuestion(Base):
    __tablename__ = "faq_questions"
    __table_args__ = (UniqueConstraint("normalized_text", name="uq_faq_questions_normalized_text"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    normalized_text: Mapped[str] = mapped_column(String(1_000), nullable=False)
    question_text: Mapped[str] = mapped_column(String(1_000), nullable=False)
    occurrence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    embedding: Mapped[list[float] | None] = mapped_column(JSONB, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(160), nullable=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class FAQQuestionEvent(Base):
    __tablename__ = "faq_question_events"

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    question_id: Mapped[str] = mapped_column(ForeignKey("faq_questions.id", ondelete="CASCADE"), index=True)
    conversation_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class FAQCandidate(Base):
    __tablename__ = "faq_candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(80), nullable=False)
    normalized_question: Mapped[str] = mapped_column(String(1_000), nullable=False)
    frequency: Mapped[int] = mapped_column(Integer, nullable=False)
    faq_score: Mapped[int] = mapped_column(Integer, nullable=False)
    priority: Mapped[str] = mapped_column(String(24), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    source_indexes: Mapped[list[int]] = mapped_column(JSONB, nullable=False)
    source_questions: Mapped[list[dict[str, object]]] = mapped_column(JSONB, nullable=False, default=list)
    source_question_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    is_published: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class FAQAnalysisState(Base):
    """Singleton row coordinating thresholded, idempotent batch analysis."""

    __tablename__ = "faq_analysis_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    last_analyzed_question_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_analyzed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lock_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


def create_database(database_url: str) -> tuple[async_sessionmaker[AsyncSession], AsyncEngine]:
    engine = create_async_engine(database_url, pool_pre_ping=True, pool_size=5, max_overflow=10)
    return async_sessionmaker(engine, expire_on_commit=False), engine
