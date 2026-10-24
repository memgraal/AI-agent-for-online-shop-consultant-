import logging

from langchain_core.messages import AIMessageChunk

from app.schemas.chat import ChatRequest
from app.services.chat import ChatService
from app.services.conversations import ConversationStore
from app.services.model_capacity import ModelOverloadedError


class ChunkedGraph:
    async def astream_events(self, _input_value, *, config, version):
        assert config["recursion_limit"] > 0
        assert version == "v2"
        for value in ("Первый фрагмент", " и второй."):
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": AIMessageChunk(content=value)},
            }


async def test_streaming_forwards_provider_chunks_without_per_character_delays(monkeypatch):
    import app.services.chat as chat_module

    monkeypatch.setattr(chat_module, "conversation_store", ConversationStore())
    service = ChatService(
        scope="customer",
        agent_factory=ChunkedGraph,
        logger=logging.getLogger("test.streaming"),
        failure_message="ошибка",
    )

    events = [event async for event in service.stream_events(ChatRequest(message="Вопрос"))]

    tokens = [payload["text"] for name, payload in events if name == "token"]
    assert tokens == ["Первый фрагмент", " и второй."]
    assert events[-1][0] == "done"


async def test_streaming_reports_retryable_overload_event(monkeypatch):
    import app.services.chat as chat_module

    monkeypatch.setattr(chat_module, "conversation_store", ConversationStore())

    class OverloadedGraph:
        async def astream_events(self, *_args, **_kwargs):
            raise ModelOverloadedError("internal queue state")
            yield  # Preserve async-generator shape.

    service = ChatService(
        scope="customer",
        agent_factory=OverloadedGraph,
        logger=logging.getLogger("test.streaming"),
        failure_message="ошибка",
    )
    events = [event async for event in service.stream_events(ChatRequest(message="Вопрос"))]

    assert events == [(
        "error",
        {
            "code": "overloaded",
            "message": "AI-сервис перегружен. Повторите запрос позже.",
        },
    )]
