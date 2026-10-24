import asyncio
import logging

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage

from app.schemas.chat import ChatRequest
from app.services.chat import ChatService
from app.services.conversations import ConversationStore


def test_conversation_store_keeps_scope_isolated_and_updates_existing_turns():
    store = ConversationStore(max_sessions=2)
    customer_id, _ = store.start_or_load("customer", None, [])
    store.append_turn("customer", customer_id, "Вопрос", "Ответ")

    with pytest.raises(HTTPException) as error:
        store.start_or_load("admin", customer_id, [])

    assert error.value.status_code == 404
    same_id, messages = store.start_or_load("customer", customer_id, [])
    assert same_id == customer_id
    assert messages == [
        {"role": "user", "content": "Вопрос"},
        {"role": "assistant", "content": "Ответ"},
    ]


def test_conversation_store_rejects_new_sessions_when_capacity_is_reached():
    store = ConversationStore(max_sessions=1)
    store.start_or_load("customer", None, [])

    with pytest.raises(HTTPException) as error:
        store.start_or_load("customer", None, [])

    assert error.value.status_code == 503
    assert error.value.headers == {"Retry-After": "5"}


def test_conversation_store_expires_oldest_sessions_before_enforcing_capacity():
    store = ConversationStore(max_sessions=1)
    old_id, _ = store.start_or_load("customer", None, [])
    store._sessions[("customer", old_id)].touched_at -= 60 * 60 * 25

    new_id, _ = store.start_or_load("customer", None, [])

    assert new_id != old_id
    assert len(store._sessions) == 1


async def test_parallel_turns_for_one_conversation_use_the_latest_completed_history(monkeypatch):
    import app.services.chat as chat_module

    store = ConversationStore(max_sessions=4)
    conversation_id, _ = store.start_or_load("customer", None, [])
    monkeypatch.setattr(chat_module, "conversation_store", store)

    class Graph:
        received = []

        async def ainvoke(self, input_value, config):
            self.received.append(input_value["messages"])
            await asyncio.sleep(0.01)
            return {"messages": [AIMessage(content="ответ") ]}

    graph = Graph()
    service = ChatService(
        scope="customer",
        agent_factory=lambda: graph,
        logger=logging.getLogger("test.conversations"),
        failure_message="ошибка",
    )

    await asyncio.gather(
        service.reply(ChatRequest(message="первый", conversation_id=conversation_id)),
        service.reply(ChatRequest(message="второй", conversation_id=conversation_id)),
    )

    assert len(graph.received) == 2
    assert "первый" in [message.content for message in graph.received[1]]
