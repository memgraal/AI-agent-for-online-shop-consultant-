"""Bounded, process-local queue used until a CRM/helpdesk adapter exists."""

import asyncio
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from app.config import settings
from app.schemas.operator import OperatorRequest, OperatorTicket
from app.services.privacy import redact_personal_data


class HandoffQueueFull(RuntimeError):
    """Raised instead of dropping a pending request when the local queue is full."""


class InMemoryOperatorQueue:
    def __init__(
        self,
        *,
        max_items: int | None = None,
        ttl_seconds: int | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._max_items = max_items if max_items is not None else settings.operator_queue_max_items
        effective_ttl = ttl_seconds if ttl_seconds is not None else settings.operator_queue_ttl_seconds
        self._ttl = timedelta(seconds=effective_ttl)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._tickets: OrderedDict[str, OperatorTicket] = OrderedDict()
        self._lock = asyncio.Lock()

    async def create(self, request: OperatorRequest) -> OperatorTicket:
        async with self._lock:
            now = self._clock()
            self._expire(now)
            if len(self._tickets) >= self._max_items:
                raise HandoffQueueFull("Временная очередь обращений заполнена.")
            ticket = OperatorTicket(
                ticket_id=uuid4().hex,
                reason=redact_personal_data(request.reason),
                conversation_id=request.conversation_id,
                created_at=now,
            )
            self._tickets[ticket.ticket_id] = ticket
            return ticket

    async def list_pending(self, *, limit: int = 100) -> list[OperatorTicket]:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        async with self._lock:
            self._expire(self._clock())
            return list(reversed(list(self._tickets.values())[-limit:]))

    def _expire(self, now: datetime) -> None:
        expiry = now - self._ttl
        while self._tickets:
            ticket_id, ticket = next(iter(self._tickets.items()))
            if ticket.created_at >= expiry:
                break
            del self._tickets[ticket_id]


operator_handoff_queue = InMemoryOperatorQueue()
