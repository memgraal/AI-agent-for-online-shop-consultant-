"""Runtime cache and PostgreSQL persistence for editable AI-agent settings."""

from copy import deepcopy
from datetime import UTC, datetime
from threading import RLock

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import settings as environment_settings
from app.constants import AI_AGENT_SETTINGS_ID
from app.prompts.admin import ADMIN_SYSTEM_PROMPT
from app.prompts.admin_catalog import ATTRIBUTE_NORMALIZATION_SYSTEM_PROMPT
from app.prompts.admin_product import PRODUCT_SEO_SYSTEM_PROMPT
from app.prompts.customer import CUSTOMER_SYSTEM_PROMPT
from app.prompts.faq_analysis import FAQ_ANALYSIS_SYSTEM_PROMPT
from app.schemas.agent_settings import AgentSettingsInput, AgentSettingsSnapshot
from app.storage.database import AssistantSettings


class AgentSettingsStore:
    """Keep settings fast to read while persisting updates transactionally."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._revision = 1
        self._updated_at = datetime.now(UTC)
        self._settings = _settings_from_environment()
        self._database_loaded = False
        self._local_override = False

    async def initialize(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """Load the singleton row after the database schema is ready."""
        async with session_factory() as session:
            row = (
                await session.execute(
                    select(AssistantSettings).where(
                        AssistantSettings.id == AI_AGENT_SETTINGS_ID
                    )
                )
            ).scalar_one_or_none()
        if row is None:
            return
        self._replace_from_row(row)

    def get(self) -> AgentSettingsSnapshot:
        with self._lock:
            # Tests and local deployments may change env-backed settings before
            # PostgreSQL is configured. Keep that fallback live until either a
            # database row or an explicit in-memory update takes ownership.
            if not self._database_loaded and not self._local_override:
                self._settings = _settings_from_environment()
            return self._snapshot()

    def prompt(self, role: str) -> str:
        """Return a prompt by role for agents that build messages per request."""
        with self._lock:
            return self._settings.system_prompts.get(role, "")

    async def update_persistent(
        self,
        new_settings: AgentSettingsInput,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> AgentSettingsSnapshot:
        """Update the singleton row under a database lock and refresh the cache."""
        async with session_factory() as session:
            row = (
                await session.execute(
                    select(AssistantSettings)
                    .where(AssistantSettings.id == AI_AGENT_SETTINGS_ID)
                    .with_for_update()
                )
            ).scalar_one()
            row.revision += 1
            _copy_to_row(row, new_settings)
            await session.commit()
            revision = row.revision
            updated_at = row.updated_at or datetime.now(UTC)
        with self._lock:
            self._revision = revision
            self._updated_at = updated_at
            self._settings = new_settings.model_copy(deep=True)
            self._database_loaded = True
            self._local_override = False
            return self._snapshot()

    def update(self, new_settings: AgentSettingsInput) -> AgentSettingsSnapshot:
        """Update the process-local cache when PostgreSQL is not configured."""
        with self._lock:
            self._revision += 1
            self._updated_at = datetime.now(UTC)
            self._settings = new_settings.model_copy(deep=True)
            self._local_override = True
            return self._snapshot()

    def _replace_from_row(self, row: AssistantSettings) -> None:
        loaded = AgentSettingsInput(
            llm_provider=row.llm_provider,
            llm_model=row.llm_model,
            llm_base_url=row.llm_base_url,
            llm_timeout_seconds=row.llm_timeout_seconds,
            llm_temperature=row.llm_temperature,
            llm_max_retries=row.llm_max_retries,
            llm_quick_max_output_tokens=row.llm_quick_max_output_tokens,
            llm_max_output_tokens=row.llm_max_output_tokens,
            chat_context_max_messages=row.chat_context_max_messages,
            chat_max_context_tokens=row.chat_max_context_tokens,
            agent_recursion_limit=row.agent_recursion_limit,
            agent_max_llm_calls=row.agent_max_llm_calls,
            system_prompts=row.system_prompts,
        )
        with self._lock:
            self._revision = row.revision
            self._updated_at = row.updated_at or datetime.now(UTC)
            self._settings = loaded
            self._database_loaded = True
            self._local_override = False

    def _snapshot(self) -> AgentSettingsSnapshot:
        return AgentSettingsSnapshot(
            revision=self._revision,
            updated_at=self._updated_at,
            settings=self._settings.model_copy(deep=True),
        )


def _settings_from_environment() -> AgentSettingsInput:
    return AgentSettingsInput(
        llm_provider=environment_settings.llm_provider,
        llm_model=environment_settings.llm_model,
        llm_base_url=environment_settings.llm_base_url,
        llm_timeout_seconds=environment_settings.llm_timeout_seconds,
        llm_temperature=environment_settings.llm_temperature,
        llm_max_retries=environment_settings.llm_max_retries,
        llm_quick_max_output_tokens=environment_settings.llm_quick_max_output_tokens,
        llm_max_output_tokens=environment_settings.llm_max_output_tokens,
        chat_context_max_messages=environment_settings.chat_context_max_messages,
        chat_max_context_tokens=environment_settings.chat_max_context_tokens,
        agent_recursion_limit=environment_settings.agent_recursion_limit,
        agent_max_llm_calls=environment_settings.agent_max_llm_calls,
        system_prompts={
            "customer": CUSTOMER_SYSTEM_PROMPT,
            "admin": ADMIN_SYSTEM_PROMPT,
            "faq_analysis": FAQ_ANALYSIS_SYSTEM_PROMPT,
            "product_seo": PRODUCT_SEO_SYSTEM_PROMPT,
            "attribute_normalization": ATTRIBUTE_NORMALIZATION_SYSTEM_PROMPT,
        },
    )


def _copy_to_row(row: AssistantSettings, value: AgentSettingsInput) -> None:
    for field in (
        "llm_provider",
        "llm_model",
        "llm_base_url",
        "llm_timeout_seconds",
        "llm_temperature",
        "llm_max_retries",
        "llm_quick_max_output_tokens",
        "llm_max_output_tokens",
        "chat_context_max_messages",
        "chat_max_context_tokens",
        "agent_recursion_limit",
        "agent_max_llm_calls",
    ):
        setattr(row, field, getattr(value, field))
    row.system_prompts = deepcopy(value.system_prompts)


agent_settings = AgentSettingsStore()
