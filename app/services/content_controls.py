"""Versioned process-local content controls; replace storage with a repository later."""

from copy import deepcopy
from datetime import UTC, datetime
import re
from threading import RLock
import unicodedata

from app.schemas.content_controls import ContentControlsInput, ContentControlsSnapshot

_TONE_INSTRUCTIONS = {
    "friendly": "Пиши доброжелательно и спокойно, без фамильярности.",
    "formal": "Пиши нейтрально и деловым тоном, сохраняя ясность.",
    "concise": "Отвечай особенно кратко, сохраняя важные условия и ограничения.",
}


class ContentControlsStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._revision = 1
        self._updated_at = datetime.now(UTC)
        self._controls = ContentControlsInput()
        self._history: list[ContentControlsSnapshot] = [self._snapshot()]

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
        sections = [f"Настройка тона: {_TONE_INSTRUCTIONS[controls.tone]}"]
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
