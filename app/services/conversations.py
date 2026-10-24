"""Process-local conversation sessions until a durable session store is selected."""

from collections import OrderedDict
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from time import monotonic
from collections.abc import AsyncIterator
from uuid import uuid4

from fastapi import HTTPException

from app.config import settings


# Retain up to 20 user/assistant turns; the model receives a smaller bounded window.
MAX_MESSAGES = 40
SESSION_TTL_SECONDS = 60 * 60 * 24


@dataclass
class Conversation:
    messages: list[dict[str, str]] = field(default_factory=list)
    touched_at: float = field(default_factory=monotonic)
    turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


class ConversationStore:
    """Temporary in-memory store. Scope prevents customer/admin session crossover."""

    def __init__(self, *, max_sessions: int | None = None) -> None:
        self._max_sessions = (
            max_sessions if max_sessions is not None else settings.chat_max_sessions
        )
        self._sessions: OrderedDict[tuple[str, str], Conversation] = OrderedDict()

    def start_or_load(
        self, scope: str, conversation_id: str | None, initial_messages: list[dict[str, str]]
    ) -> tuple[str, list[dict[str, str]]]:
        self._expire_old_sessions()
        if conversation_id is None:
            if len(self._sessions) >= self._max_sessions:
                raise HTTPException(
                    status_code=503,
                    detail="Сервис временно не может создать новую сессию. Повторите позже.",
                    headers={"Retry-After": "5"},
                )
            conversation_id = str(uuid4())
            self._sessions[(scope, conversation_id)] = Conversation(
                messages=initial_messages[-MAX_MESSAGES:]
            )
        session = self._sessions.get((scope, conversation_id))
        if session is None:
            raise HTTPException(status_code=404, detail="Сессия не найдена или истекла.")
        session.touched_at = monotonic()
        self._sessions.move_to_end((scope, conversation_id))
        return conversation_id, list(session.messages)

    def append_turn(self, scope: str, conversation_id: str, user_text: str, answer: str) -> None:
        session = self._sessions.get((scope, conversation_id))
        if session is None:
            raise HTTPException(status_code=404, detail="Сессия не найдена или истекла.")
        session.messages.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer},
        ])
        session.messages = session.messages[-MAX_MESSAGES:]
        session.touched_at = monotonic()
        self._sessions.move_to_end((scope, conversation_id))

    @asynccontextmanager
    async def serialize_turn(self, scope: str, conversation_id: str) -> AsyncIterator[None]:
        session = self._sessions.get((scope, conversation_id))
        if session is None:
            raise HTTPException(status_code=404, detail="Сессия не найдена или истекла.")
        await session.turn_lock.acquire()
        try:
            if self._sessions.get((scope, conversation_id)) is not session:
                raise HTTPException(status_code=404, detail="Сессия не найдена или истекла.")
            session.touched_at = monotonic()
            self._sessions.move_to_end((scope, conversation_id))
            yield
        finally:
            session.turn_lock.release()

    def _expire_old_sessions(self) -> None:
        now = monotonic()
        # Ordered by most recent touch; expired sessions form a contiguous prefix.
        while self._sessions:
            key, session = next(iter(self._sessions.items()))
            if now - session.touched_at <= SESSION_TTL_SECONDS:
                break
            del self._sessions[key]


conversation_store = ConversationStore()
