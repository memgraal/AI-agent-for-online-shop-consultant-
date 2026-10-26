"""Shared orchestration for customer and administrator assistant conversations."""

import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from fastapi import HTTPException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.schemas.chat import ChatRequest
from app.config import settings
from app.services.conversations import conversation_store
from app.services.logging import log_event
from app.services.model_capacity import ModelOverloadedError
from app.services.privacy import redact_personal_data
from app.services.token_budget import estimate_message_tokens


STREAM_PROGRESS = {
    "search_catalog": "Ищу подходящие товары",
    "compare_products": "Сравниваю характеристики",
    "search_knowledge_base": "Проверяю документацию",
    "calculate_order_total": "Проверяю расчёт корзины",
    "create_installation_brief": "Собираю ТЗ по объекту",
    "call_operator": "Готовлю обращение оператору",
}


@dataclass(frozen=True)
class ChatResult:
    answer: str
    conversation_id: str


class ChatService:
    def __init__(
        self,
        *,
        scope: str,
        agent_factory: Callable[[], Any],
        logger: logging.Logger,
        failure_message: str,
    ) -> None:
        self.scope = scope
        self.agent_factory = agent_factory
        self.logger = logger
        self.failure_message = failure_message

    def _prepare(self, request: ChatRequest) -> tuple[str, str, list[BaseMessage]]:
        user_text = redact_personal_data(request.message)
        user_message = HumanMessage(content=user_text)
        context_limit = settings.chat_max_context_tokens
        user_tokens = estimate_message_tokens([user_message], settings.llm_model)
        if user_tokens > context_limit:
            raise HTTPException(
                status_code=413,
                detail=f"Сообщение превышает лимит контекста ({context_limit} токенов). Сократите текст.",
            )

        history_seed = [
            {"role": item.role, "content": redact_personal_data(item.content)}
            for item in request.history
        ]
        conversation_id, history = conversation_store.start_or_load(
            self.scope, request.conversation_id, history_seed
        )
        history = history[-settings.chat_context_max_messages :]
        messages: list[BaseMessage] = [
            HumanMessage(content=item["content"])
            if item["role"] == "user"
            else AIMessage(content=item["content"])
            for item in history
        ]
        messages.append(user_message)
        dropped_for_tokens = 0
        while (
            len(messages) > 1
            and estimate_message_tokens(messages, settings.llm_model) > context_limit
        ):
            messages.pop(0)
            dropped_for_tokens += 1
        log_event(
            self.logger,
            logging.INFO,
            "chat.context_prepared",
            agent=self.scope,
            context_messages=len(messages),
            context_tokens=estimate_message_tokens(messages, settings.llm_model),
            context_messages_dropped=dropped_for_tokens,
            context_token_limit=context_limit,
        )
        return conversation_id, user_text, messages

    def _log_started(self, request: ChatRequest, *, streaming: bool = False) -> float:
        started = perf_counter()
        log_event(
            self.logger,
            logging.INFO,
            "chat.stream_started" if streaming else "chat.started",
            agent=self.scope,
            history_count=len(request.history),
            message_chars=len(request.message),
            detail=request.detail,
        )
        return started

    async def reply(self, request: ChatRequest) -> ChatResult:
        started = self._log_started(request)
        try:
            conversation_id, _, _ = self._prepare(request)
            turn_request = request.model_copy(
                update={"conversation_id": conversation_id}
            )
            async with conversation_store.serialize_turn(self.scope, conversation_id):
                conversation_id, user_text, messages = self._prepare(turn_request)
                result = await self.agent_factory().ainvoke(
                    {"messages": messages, "detail": request.detail, "llm_calls": 0},
                    config={"recursion_limit": settings.agent_recursion_limit},
                )
                answer = self._message_text(result["messages"][-1].content)
                if not answer:
                    answer = self._empty_answer_fallback()
                conversation_store.append_turn(
                    self.scope, conversation_id, user_text, answer
                )

            log_event(
                self.logger,
                logging.INFO,
                "chat.completed",
                agent=self.scope,
                answer_chars=len(answer),
                duration_ms=self._duration(started),
            )
            return ChatResult(answer=answer, conversation_id=conversation_id)
        except HTTPException:
            raise
        except Exception as exc:
            self._log_failure("chat.failed", started, exc)
            raise

    def stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """Format shared stream events as SSE for the legacy HTTP endpoint."""
        events = self.stream_events(request)

        async def sse_events() -> AsyncIterator[str]:
            async for event_name, payload in events:
                yield self._sse(event_name, payload)

        return sse_events()

    def stream_events(
        self,
        request: ChatRequest,
    ) -> AsyncIterator[tuple[str, dict[str, str]]]:
        """Stream typed events independently of the transport protocol."""
        # Prepare eagerly so invalid/expired session IDs can be reported by the caller.
        conversation_id, _, _ = self._prepare(request)
        turn_request = request.model_copy(update={"conversation_id": conversation_id})
        graph = self.agent_factory()
        config = {"recursion_limit": settings.agent_recursion_limit}

        async def events() -> AsyncIterator[tuple[str, dict[str, str]]]:
            started = self._log_started(request, streaming=True)
            try:
                async with conversation_store.serialize_turn(
                    self.scope, conversation_id
                ):
                    _, user_text, messages = self._prepare(turn_request)
                    streamed_answer = ""
                    async for item in graph.astream_events(
                        {
                            "messages": messages,
                            "detail": request.detail,
                            "llm_calls": 0,
                        },
                        config=config,
                        version="v2",
                    ):
                        event_name = item.get("event")
                        if event_name == "on_chat_model_stream":
                            text = self._chunk_text(item.get("data", {}).get("chunk"))
                            if text:
                                streamed_answer += text
                                yield "token", {"text": text}
                        elif event_name == "on_tool_start":
                            progress = STREAM_PROGRESS.get(item.get("name", ""))
                            if progress:
                                yield "progress", {"text": progress}

                    if not streamed_answer:
                        streamed_answer = self._empty_answer_fallback()
                        yield "token", {"text": streamed_answer}
                    conversation_store.append_turn(
                        self.scope, conversation_id, user_text, streamed_answer
                    )
                yield "done", {"conversation_id": conversation_id}
                log_event(
                    self.logger,
                    logging.INFO,
                    "chat.stream_completed",
                    agent=self.scope,
                    duration_ms=self._duration(started),
                )
            except ModelOverloadedError as exc:
                self._log_failure("chat.stream_failed", started, exc)
                yield (
                    "error",
                    {
                        "code": "overloaded",
                        "message": "AI-сервис перегружен. Повторите запрос позже.",
                    },
                )
            except Exception as exc:
                self._log_failure("chat.stream_failed", started, exc)
                yield "error", {"message": self.failure_message}

        return events()

    def _log_failure(self, event: str, started: float, exc: Exception) -> None:
        log_event(
            self.logger,
            logging.ERROR,
            event,
            agent=self.scope,
            duration_ms=self._duration(started),
            error_type=type(exc).__name__,
        )

    def _empty_answer_fallback(self) -> str:
        log_event(
            self.logger, logging.WARNING, "chat.empty_model_response", agent=self.scope
        )
        return "Не получилось сформировать ответ. Попробуйте задать вопрос ещё раз."

    @staticmethod
    def _message_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            ]
            return "".join(parts).strip()
        return ""

    @staticmethod
    def _duration(started: float) -> float:
        return round((perf_counter() - started) * 1000, 2)

    @staticmethod
    def _chunk_text(chunk: object) -> str:
        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return ""

    @staticmethod
    def _sse(event: str, payload: dict[str, str]) -> str:
        data = json.dumps(payload, ensure_ascii=False)
        return f"event: {event}\ndata: {data}\n\n"
