import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api import chat as chat_api
from app.config import settings
from app.main import app
from app.services.chat import ChatResult
from app.services.rate_limit import rate_limiter


class FakeChatService:
    failure_message = "ошибка"

    async def reply(self, request):
        return ChatResult(answer=f"Ответ: {request.message}", conversation_id="conversation-1")

    async def stream_events(self, request):
        yield "token", {"text": f"Токен: {request.message}"}
        yield "done", {"conversation_id": "conversation-1"}


def test_websocket_supports_full_and_streamed_turns_on_same_connection(monkeypatch):
    monkeypatch.setattr(settings, "allowed_origins", "https://shop.example")
    monkeypatch.setattr(chat_api, "_service", lambda: FakeChatService())

    with TestClient(app) as client:
        with client.websocket_connect(
            "/api/chat/ws", headers={"origin": "https://shop.example"}
        ) as websocket:
            websocket.send_json({"request_id": "turn-1", "message": "вопрос"})
            result = websocket.receive_json()
            assert result == {
                "type": "result",
                "request_id": "turn-1",
                "answer": "Ответ: вопрос",
                "conversation_id": "conversation-1",
            }

            websocket.send_json({"request_id": "turn-2", "message": "уточнение", "stream": True})
            token = websocket.receive_json()
            done = websocket.receive_json()
            assert token == {"type": "token", "request_id": "turn-2", "text": "Токен: уточнение"}
            assert done == {
                "type": "done",
                "request_id": "turn-2",
                "conversation_id": "conversation-1",
            }


def test_websocket_closes_for_unlisted_origin(monkeypatch):
    monkeypatch.setattr(settings, "allowed_origins", "https://shop.example")

    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as error:
            with client.websocket_connect(
                "/api/chat/ws", headers={"origin": "https://attacker.example"}
            ):
                pass

    assert error.value.code == 1008


async def test_websocket_frames_share_the_configured_rate_limit(monkeypatch):
    await rate_limiter.clear()
    monkeypatch.setattr(settings, "allowed_origins", "https://shop.example")
    monkeypatch.setattr(settings, "api_rate_limit_max_requests", 1)
    monkeypatch.setattr(chat_api, "_service", lambda: FakeChatService())

    try:
        with TestClient(app) as client:
            with client.websocket_connect(
                "/api/chat/ws", headers={"origin": "https://shop.example"}
            ) as websocket:
                websocket.send_json({"request_id": "first", "message": "первый"})
                assert websocket.receive_json()["type"] == "result"
                websocket.send_json({"request_id": "second", "message": "второй"})
                error = websocket.receive_json()

        assert error["type"] == "error"
        assert error["code"] == "rate_limited"
        assert error["request_id"] == "second"
    finally:
        await rate_limiter.clear()
