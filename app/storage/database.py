"""Small async SQLAlchemy storage layer for assistant runtime state."""

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, Text, func, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.constants import AI_AGENT_SETTINGS_ID


class Base(DeclarativeBase):
    pass


class AssistantContentState(Base):
    __tablename__ = "ai_assistant_content_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    controls: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class AssistantContentHistory(Base):
    __tablename__ = "ai_assistant_content_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    controls: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class AssistantSettings(Base):
    """Singleton configuration record for mutable AI-agent settings.

    Provider secrets intentionally stay in environment variables or a secret
    manager. This table stores non-secret runtime settings and editable prompts
    so the future admin API can update them without changing the deployment.
    """

    __tablename__ = "ai_agent_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=AI_AGENT_SETTINGS_ID)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    llm_provider: Mapped[str] = mapped_column(Text, nullable=False)
    llm_model: Mapped[str] = mapped_column(Text, nullable=False)
    llm_base_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_timeout_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    llm_temperature: Mapped[float] = mapped_column(Float, nullable=False)
    llm_max_retries: Mapped[int] = mapped_column(Integer, nullable=False)
    llm_quick_max_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    llm_max_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    chat_context_max_messages: Mapped[int] = mapped_column(Integer, nullable=False)
    chat_max_context_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    agent_recursion_limit: Mapped[int] = mapped_column(Integer, nullable=False)
    agent_max_llm_calls: Mapped[int] = mapped_column(Integer, nullable=False)
    system_prompts: Mapped[dict[str, str]] = mapped_column(JSONB, nullable=False, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


def create_database(url: str) -> tuple[async_sessionmaker[AsyncSession], AsyncEngine]:
    engine = create_async_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)
    return async_sessionmaker(engine, expire_on_commit=False), engine


async def initialize_database(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        # ``create_all`` does not alter an existing table. This keeps databases
        # created by the first settings-table version compatible after adding
        # the context-message limit.
        await connection.execute(
            text(
                "ALTER TABLE ai_agent_settings "
                "ADD COLUMN IF NOT EXISTS chat_context_max_messages "
                "INTEGER NOT NULL DEFAULT 20"
            )
        )
    await _seed_agent_settings(engine)


async def _seed_agent_settings(engine: AsyncEngine) -> None:
    """Create the initial agent settings row without persisting any secret."""
    from app.config import settings
    from app.prompts.admin import ADMIN_SYSTEM_PROMPT
    from app.prompts.admin_catalog import ATTRIBUTE_NORMALIZATION_SYSTEM_PROMPT
    from app.prompts.admin_product import PRODUCT_SEO_SYSTEM_PROMPT
    from app.prompts.customer import CUSTOMER_SYSTEM_PROMPT
    from app.prompts.faq_analysis import FAQ_ANALYSIS_SYSTEM_PROMPT

    default_prompts = _default_system_prompts(
        customer=CUSTOMER_SYSTEM_PROMPT,
        admin=ADMIN_SYSTEM_PROMPT,
        faq_analysis=FAQ_ANALYSIS_SYSTEM_PROMPT,
        product_seo=PRODUCT_SEO_SYSTEM_PROMPT,
        attribute_normalization=ATTRIBUTE_NORMALIZATION_SYSTEM_PROMPT,
    )
    async with AsyncSession(engine, expire_on_commit=False) as session:
        existing = (
            await session.execute(
                select(AssistantSettings).where(
                    AssistantSettings.id == AI_AGENT_SETTINGS_ID
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            # Preserve edited prompts, but backfill roles added in a later
            # application version so old rows remain fully usable.
            existing_prompts = existing.system_prompts or {}
            missing_prompts = {
                key: value
                for key, value in default_prompts.items()
                if key not in existing_prompts
            }
            if missing_prompts:
                existing.system_prompts = {
                    **existing_prompts,
                    **missing_prompts,
                }
                await session.commit()
            return

        session.add(
            AssistantSettings(
                id=AI_AGENT_SETTINGS_ID,
                revision=1,
                llm_provider=settings.llm_provider.value,
                llm_model=settings.llm_model,
                llm_base_url=settings.llm_base_url,
                llm_timeout_seconds=settings.llm_timeout_seconds,
                llm_temperature=settings.llm_temperature,
                llm_max_retries=settings.llm_max_retries,
                llm_quick_max_output_tokens=settings.llm_quick_max_output_tokens,
                llm_max_output_tokens=settings.llm_max_output_tokens,
                chat_context_max_messages=settings.chat_context_max_messages,
                chat_max_context_tokens=settings.chat_max_context_tokens,
                agent_recursion_limit=settings.agent_recursion_limit,
                agent_max_llm_calls=settings.agent_max_llm_calls,
                system_prompts=default_prompts,
            )
        )
        await session.commit()


def _default_system_prompts(
    *,
    customer: str,
    admin: str,
    faq_analysis: str,
    product_seo: str,
    attribute_normalization: str,
) -> dict[str, str]:
    """Return the initial role-to-prompt mapping stored in PostgreSQL.

    The keys are application role identifiers, not model names. Keeping this
    mapping in JSONB lets a future admin API add or edit a prompt without
    changing the table schema for every new agent role.
    """
    return {
        "customer": customer,
        "admin": admin,
        "faq_analysis": faq_analysis,
        "product_seo": product_seo,
        "attribute_normalization": attribute_normalization,
    }


async def close_database(engine: AsyncEngine | None) -> None:
    if engine is not None:
        await engine.dispose()
