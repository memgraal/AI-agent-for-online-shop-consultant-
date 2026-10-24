from datetime import UTC, datetime, timedelta

import pytest

from app.schemas.operator import OperatorRequest
from app.services.operator_handoff import HandoffQueueFull, InMemoryOperatorQueue


async def test_queue_creates_local_ticket_and_masks_identifiers():
    queue = InMemoryOperatorQueue()
    ticket = await queue.create(OperatorRequest(
        reason="Нужна помощь, мой ИНН 7707083893, телефон +7 (900) 123-45-67",
        conversation_id="conversation-1",
    ))

    assert ticket.status == "queued_local"
    assert ticket.conversation_id == "conversation-1"
    assert "7707083893" not in ticket.reason
    assert "+7 (900) 123-45-67" not in ticket.reason
    assert ticket in await queue.list_pending()


async def test_queue_rejects_when_full_without_dropping_existing_ticket():
    queue = InMemoryOperatorQueue(max_items=1)
    first = await queue.create(OperatorRequest(reason="Перезвоните по заказу"))

    with pytest.raises(HandoffQueueFull):
        await queue.create(OperatorRequest(reason="Нужна консультация"))

    assert await queue.list_pending() == [first]


async def test_queue_expires_old_items():
    current_time = datetime(2026, 10, 24, tzinfo=UTC)
    queue = InMemoryOperatorQueue(
        ttl_seconds=60,
        clock=lambda: current_time,
    )
    ticket = await queue.create(OperatorRequest(reason="Помощь с выбором"))
    current_time += timedelta(seconds=61)

    assert await queue.list_pending() == []
    assert ticket.status == "queued_local"
