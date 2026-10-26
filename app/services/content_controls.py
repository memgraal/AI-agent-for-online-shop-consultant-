"""Versioned assistant content controls with a database-backed cache."""

from copy import deepcopy
from datetime import UTC, datetime
import re
from threading import RLock
import unicodedata

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.constants import TONE_INSTRUCTIONS
from app.schemas.content_controls import ContentControlsInput, ContentControlsSnapshot
from app.storage.database import AssistantContentHistory, AssistantContentState

class ContentControlsStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._revision = 1
        self._updated_at = datetime.now(UTC)
        self._controls = ContentControlsInput()
        self._history: list[ContentControlsSnapshot] = [self._snapshot()]

    async def initialize(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        """Load persisted controls once during application startup."""
        async with session_factory() as session:
            state = (await session.execute(
                select(AssistantContentState).where(AssistantContentState.id == 1)
            )).scalar_one_or_none()
            if state is None:
                state = AssistantContentState(
                    id=1,
                    revision=1,
                    controls=ContentControlsInput().model_dump(mode="json"),
                )
                session.add(state)
                session.add(AssistantContentHistory(
                    revision=1,
                    controls=state.controls,
                ))
                await session.commit()
                return

            controls = ContentControlsInput.model_validate(state.controls)
            history_rows = (await session.execute(
                select(AssistantContentHistory)
                .order_by(desc(AssistantContentHistory.revision))
                .limit(50)
            )).scalars().all()
        with self._lock:
            self._revision = state.revision
            self._updated_at = state.updated_at
            self._controls = controls
            self._history = [
                ContentControlsSnapshot(
                    revision=row.revision,
                    updated_at=row.updated_at,
                    controls=ContentControlsInput.model_validate(row.controls),
                )
                for row in reversed(history_rows)
            ] or [self._snapshot()]

    async def update_persistent(
        self,
        controls: ContentControlsInput,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> ContentControlsSnapshot:
        """Persist an update transactionally, then refresh the in-process cache."""
        async with session_factory() as session:
            state = (await session.execute(
                select(AssistantContentState)
                .where(AssistantContentState.id == 1)
                .with_for_update()
            )).scalar_one()
            state.revision += 1
            state.controls = controls.model_dump(mode="json")
            history = AssistantContentHistory(
                revision=state.revision,
                controls=state.controls,
            )
            session.add(history)
            await session.commit()
            updated_at = state.updated_at
            revision = state.revision

        with self._lock:
            self._revision = revision
            self._updated_at = updated_at
            self._controls = controls.model_copy(deep=True)
            snapshot = self._snapshot()
            self._history.append(snapshot)
            del self._history[:-50]
            return snapshot

    def get(self) -> ContentControlsSnapshot:
        with self._lock:
            return self._snapshot()

    def update(self, controls: ContentControlsInput) -> ContentControlsSnapshot:
        with self._lock:
            self._revision += 1
            self._updated_at = datetime.now(UTC)
            self._controls = controls.model_copy(deep=True)
            snapshot = self._snapshot()
            self._history.append(snapshot)
            del self._history[:-50]
            return snapshot

    def history(self) -> list[ContentControlsSnapshot]:
        with self._lock:
            return deepcopy(self._history)

    def prompt_overlay(self) -> str:
        with self._lock:
            controls = self._controls.model_copy(deep=True)
        sections = [f"Настройка тона: {TONE_INSTRUCTIONS[controls.tone.value]}"]
        active_faqs = [faq for faq in controls.faqs if faq.enabled]
        if active_faqs:
            sections.append(
                "Проверенные администратором ответы FAQ (используй только при точном совпадении):\n"
                + "\n".join(
                    f"Вопрос: {faq.question}\nОтвет: {faq.answer}"
                    for faq in active_faqs
                )
            )
        return "\n\n".join(sections)

    def disabled_topic_response(self, message: str) -> str | None:
        normalized_message = _normalize(message)
        with self._lock:
            controls = self._controls.model_copy(deep=True)
        for topic in controls.disabled_topics:
            if any(_contains_term(normalized_message, _normalize(term)) for term in topic.match_terms):
                return controls.disabled_topic_message
        return None

    def _snapshot(self) -> ContentControlsSnapshot:
        return ContentControlsSnapshot(
            revision=self._revision,
            updated_at=self._updated_at,
            controls=self._controls.model_copy(deep=True),
        )


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")


def _contains_term(text: str, term: str) -> bool:
    text_words = _word_stems(text)
    term_words = _word_stems(term)
    if not term_words or len(term_words) > len(text_words):
        return False
    return any(
        text_words[index:index + len(term_words)] == term_words
        for index in range(len(text_words) - len(term_words) + 1)
    )


def _word_stems(value: str) -> list[str]:
    suffixes = (
        "иями", "ями", "ами", "ого", "ему", "ому", "ыми", "ими", "ую", "юю",
        "ая", "яя", "ое", "ее", "ые", "ие", "ой", "ей", "ым", "им", "ом", "ем",
        "ах", "ях", "ов", "ев", "ей", "ам", "ям", "а", "я", "ы", "и", "е", "о", "у", "ю",
    )
    words = re.findall(r"\w+", _normalize(value), flags=re.UNICODE)
    stems: list[str] = []
    for word in words:
        for suffix in suffixes:
            if len(word) - len(suffix) >= 3 and word.endswith(suffix):
                word = word[:-len(suffix)]
                break
        stems.append(word)
    return stems


content_controls = ContentControlsStore()
