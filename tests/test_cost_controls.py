import logging

import pytest
from fastapi import HTTPException

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.common import model_node
from app.api.schemas import ChatRequest
from app.config import settings
from app.services import chat as chat_service_module
from app.services.chat import ChatService
from app.services.conversations import ConversationStore


class FakeModel:
    def __init__(self):
        self.kwargs = None

    async def ainvoke(self, messages, **kwargs):
        self.kwargs = kwargs
        return AIMessage(content="Готово", usage_metadata={
            "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
        })


async def test_model_node_applies_lower_quick_output_limit(monkeypatch):
    monkeypatch.setattr(settings, "llm_quick_max_output_tokens", 320)
    model = FakeModel()
    node = model_node(model, "system prompt")

    await node({"messages": [HumanMessage(content="Вопрос")], "detail": "quick"})

    assert model.kwargs["max_tokens"] == 320


async def test_model_node_uses_detailed_output_limit(monkeypatch):
    monkeypatch.setattr(settings, "llm_max_output_tokens", 1100)
    model = FakeModel()
    node = model_node(model, "system prompt")

    await node({"messages": [HumanMessage(content="Вопрос")], "detail": "detailed"})

    assert model.kwargs["max_tokens"] == 1100


async def test_chat_service_limits_context_sent_to_model(monkeypatch):
    monkeypatch.setattr(chat_service_module, "conversation_store", ConversationStore())
    monkeypatch.setattr(settings, "chat_context_max_messages", 4)

    class FakeGraph:
        received = None

        async def ainvoke(self, input_value, config):
            self.received = input_value
            return {"messages": [AIMessage(content="Ответ") ]}

    graph = FakeGraph()
    service = ChatService(
        scope="customer", agent_factory=lambda: graph,
        logger=logging.getLogger("test"), failure_message="ошибка",
    )
    history = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"history-{index}"}
        for index in range(10)
    ]

    await service.reply(ChatRequest(message="текущий вопрос", history=history))

    sent = graph.received["messages"]
    assert [message.content for message in sent] == [
        "history-6", "history-7", "history-8", "history-9", "текущий вопрос",
    ]


async def test_chat_service_drops_old_messages_to_fit_token_budget(monkeypatch):
    monkeypatch.setattr(chat_service_module, "conversation_store", ConversationStore())
    monkeypatch.setattr(settings, "chat_context_max_messages", 12)
    monkeypatch.setattr(settings, "chat_max_context_tokens", 300)

    class FakeGraph:
        received = None

        async def ainvoke(self, input_value, config):
            self.received = input_value
            return {"messages": [AIMessage(content="Ответ")]}

    graph = FakeGraph()
    service = ChatService(scope="customer", agent_factory=lambda: graph,
                          logger=logging.getLogger("test"), failure_message="ошибка")
    history = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"old-{index} " + "x " * 220}
        for index in range(8)
    ]

    await service.reply(ChatRequest(message="последний вопрос", history=history))

    contents = [message.content for message in graph.received["messages"]]
    assert contents[-1] == "последний вопрос"
    assert len(contents) < len(history) + 1
    assert "old-0" not in " ".join(contents)


async def test_chat_service_rejects_current_message_over_token_budget(monkeypatch):
    monkeypatch.setattr(chat_service_module, "conversation_store", ConversationStore())
    monkeypatch.setattr(settings, "chat_max_context_tokens", 256)
    service = ChatService(scope="customer", agent_factory=lambda: None,
                          logger=logging.getLogger("test"), failure_message="ошибка")

    with pytest.raises(HTTPException) as error:
        await service.reply(ChatRequest(message="слово " * 500))

    assert error.value.status_code == 413
