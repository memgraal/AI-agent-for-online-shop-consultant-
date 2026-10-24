import json
import logging
import re
from functools import lru_cache
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from app.agents.customer import build_customer_agent
from app.config import settings
from app.api.chat_handlers import invoke_chat, stream_chat
from app.schemas.chat import ChatRequest, ChatResponse, WebSocketChatRequest
from app.services.chat import ChatService
from app.services.logging import log_event, request_id_context
from app.services.model_capacity import ModelOverloadedError
from app.services.rate_limit import rate_limiter

router = APIRouter(prefix="/api/chat", tags=["customer chat"])
logger = logging.getLogger("app.api.customer_chat")


@lru_cache(maxsize=1)
def _agent():
    return build_customer_agent()


@lru_cache(maxsize=1)
def _service() -> ChatService:
    return ChatService(
        scope="customer",
        agent_factory=lambda: _agent(),
        logger=logger,
        failure_message="Не удалось получить ответ ИИ-сервиса.",
    )


@router.post("", response_model=ChatResponse)
async def customer_chat(request: ChatRequest) -> ChatResponse:
    return await invoke_chat(_service(), request)


@router.post("/stream", response_class=StreamingResponse)
async def customer_chat_stream(request: ChatRequest) -> StreamingResponse:
    return stream_chat(_service(), request)


def _request_id(value: object) -> str:
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9._-]{1,64}", value):
        return value
    return str(uuid4())


async def _send_ws_error(
    websocket: WebSocket,
    request_id: str,
    *,
    code: str,
    message: str,
) -> None:
    await websocket.send_json({
        "type": "error",
        "request_id": request_id,
        "code": code,
        "message": message,
    })


async def _send_stream_events(
    websocket: WebSocket,
    service: ChatService,
    request: WebSocketChatRequest,
    request_id: str,
    active_conversation_id: str | None,
) -> tuple[str | None, bool]:
    """Forward service events as JSON and return the conversation status."""
    completed = False
    async for event_name, event_payload in service.stream_events(request):
        if event_name == "done":
            active_conversation_id = event_payload.get(
                "conversation_id",
                active_conversation_id,
            )
            completed = True

        payload = {
            "type": event_name,
            "request_id": request_id,
            **event_payload,
        }
        if event_name == "error":
            payload["code"] = event_payload.get("code", "request_failed")
        await websocket.send_json(payload)

        if event_name == "error":
            break

    return active_conversation_id, completed


async def _handle_ws_request(
    websocket: WebSocket,
    payload: object,
    active_conversation_id: str | None,
) -> str | None:
    """Validate and process one WebSocket frame, preserving its trace ID."""
    raw_request_id = (
        payload.get("request_id")
        if isinstance(payload, dict)
        else None
    )
    request_id = _request_id(raw_request_id)
    request_token = request_id_context.set(request_id)
    started = perf_counter()

    try:
        try:
            request = WebSocketChatRequest.model_validate(payload)
        except ValidationError as exc:
            log_event(
                logger,
                logging.WARNING,
                "chat.websocket_invalid_request",
                agent="customer",
                error_type=type(exc).__name__,
            )
            await _send_ws_error(
                websocket,
                request_id,
                code="invalid_request",
                message="Проверьте формат сообщения и обязательные поля.",
            )
            return active_conversation_id

        if active_conversation_id is not None:
            if request.conversation_id not in (None, active_conversation_id):
                await _send_ws_error(
                    websocket,
                    request_id,
                    code="conversation_mismatch",
                    message=(
                        "Это WebSocket-соединение уже привязано к другому "
                        "диалогу. Откройте новое соединение."
                    ),
                )
                return active_conversation_id
            request.conversation_id = active_conversation_id

        log_event(
            logger,
            logging.INFO,
            "chat.websocket_request_started",
            agent="customer",
            history_count=len(request.history),
            message_chars=len(request.message),
            detail=request.detail,
        )
        service = _service()

        if request.stream:
            active_conversation_id, completed = await _send_stream_events(
                websocket,
                service,
                request,
                request_id,
                active_conversation_id,
            )
            event = (
                "chat.websocket_request_completed"
                if completed
                else "chat.websocket_request_failed"
            )
            log_event(
                logger,
                logging.INFO if completed else logging.ERROR,
                event,
                agent="customer",
                duration_ms=round((perf_counter() - started) * 1000, 2),
                error_type=None if completed else "StreamError",
            )
            return active_conversation_id

        result = await service.reply(request)
        active_conversation_id = result.conversation_id
        await websocket.send_json({
            "type": "result",
            "request_id": request_id,
            "answer": result.answer,
            "conversation_id": result.conversation_id,
        })
        log_event(
            logger,
            logging.INFO,
            "chat.websocket_request_completed",
            agent="customer",
            answer_chars=len(result.answer),
            duration_ms=round((perf_counter() - started) * 1000, 2),
        )
        return active_conversation_id
    except HTTPException as exc:
        log_event(
            logger,
            logging.WARNING,
            "chat.websocket_request_failed",
            agent="customer",
            status_code=exc.status_code,
            duration_ms=round((perf_counter() - started) * 1000, 2),
            error_type=type(exc).__name__,
        )
        code = (
            "conversation_not_found"
            if exc.status_code == 404
            else "request_failed"
        )
        await _send_ws_error(
            websocket,
            request_id,
            code=code,
            message=(
                str(exc.detail)
                if isinstance(exc.detail, str)
                else _service().failure_message
            ),
        )
        return active_conversation_id
    except ModelOverloadedError as exc:
        log_event(
            logger,
            logging.WARNING,
            "chat.websocket_request_overloaded",
            agent="customer",
            duration_ms=round((perf_counter() - started) * 1000, 2),
            error_type=type(exc).__name__,
        )
        await _send_ws_error(
            websocket,
            request_id,
            code="overloaded",
            message="AI-сервис перегружен. Повторите запрос позже.",
        )
        return active_conversation_id
    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "chat.websocket_request_failed",
            agent="customer",
            duration_ms=round((perf_counter() - started) * 1000, 2),
            error_type=type(exc).__name__,
        )
        await _send_ws_error(
            websocket,
            request_id,
            code="request_failed",
            message=_service().failure_message,
        )
        return active_conversation_id
    finally:
        request_id_context.reset(request_token)


@router.websocket("/ws")
async def customer_chat_websocket(websocket: WebSocket) -> None:
    """Handle full-response and token-stream chat frames on one connection.

    Client frames follow ``WebSocketChatRequest``. Responses are JSON objects
    with a ``type`` field: ``result`` for a full answer, or ``progress``,
    ``token`` and ``done`` for streaming. Errors use ``type=error``.
    """
    origin = websocket.headers.get("origin", "").rstrip("/")
    if not settings.cors_origins or origin not in settings.cors_origins:
        log_event(
            logger,
            logging.WARNING,
            "chat.websocket_origin_rejected",
        )
        await websocket.close(code=1008, reason="Origin is not allowed")
        return
    await websocket.accept()
    connection_id = str(uuid4())
    connection_token = request_id_context.set(connection_id)
    active_conversation_id: str | None = None
    log_event(
        logger,
        logging.INFO,
        "chat.websocket_connected",
        agent="customer",
    )

    try:
        while True:
            try:
                payload = await websocket.receive_json()
            except json.JSONDecodeError:
                await _send_ws_error(
                    websocket,
                    str(uuid4()),
                    code="invalid_json",
                    message="Сообщение должно быть корректным JSON.",
                )
                continue
            client_key = websocket.client.host if websocket.client else "unknown"
            allowed, retry_after = await rate_limiter.check(client_key)
            if not allowed:
                raw_request_id = payload.get("request_id") if isinstance(payload, dict) else None
                await _send_ws_error(
                    websocket,
                    _request_id(raw_request_id),
                    code="rate_limited",
                    message=f"Слишком много запросов. Повторите через {retry_after} сек.",
                )
                continue
            active_conversation_id = await _handle_ws_request(
                websocket,
                payload,
                active_conversation_id,
            )
    except WebSocketDisconnect:
        log_event(
            logger,
            logging.INFO,
            "chat.websocket_disconnected",
            agent="customer",
        )
    finally:
        request_id_context.reset(connection_token)
