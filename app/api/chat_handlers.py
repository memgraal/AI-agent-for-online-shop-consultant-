"""Translate shared chat service results and errors into HTTP responses."""

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat import ChatService
from app.services.model_capacity import ModelOverloadedError


async def invoke_chat(service: ChatService, request: ChatRequest) -> ChatResponse:
    try:
        result = await service.reply(request)
        return ChatResponse(answer=result.answer, conversation_id=result.conversation_id)
    except HTTPException:
        raise
    except ModelOverloadedError as exc:
        raise HTTPException(
            status_code=503,
            detail="AI-сервис перегружен. Повторите запрос позже.",
            headers={"Retry-After": "5"},
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=service.failure_message) from exc


def stream_chat(service: ChatService, request: ChatRequest) -> StreamingResponse:
    try:
        events = service.stream(request)
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=service.failure_message) from exc

    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
