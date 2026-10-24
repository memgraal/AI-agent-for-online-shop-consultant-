"""Per-process backpressure for requests to external language model providers."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.config import settings


class ModelOverloadedError(RuntimeError):
    """Raised when the provider queue stays full beyond its bounded wait."""


class ModelCapacityLimiter:
    def __init__(
        self,
        *,
        max_concurrent_requests: int,
        queue_timeout_seconds: float,
    ) -> None:
        if max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be positive")
        if queue_timeout_seconds <= 0:
            raise ValueError("queue_timeout_seconds must be positive")
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._queue_timeout_seconds = queue_timeout_seconds

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self._queue_timeout_seconds,
            )
        except TimeoutError as exc:
            raise ModelOverloadedError(
                "AI-сервис перегружен. Повторите запрос позже."
            ) from exc
        try:
            yield
        finally:
            self._semaphore.release()


model_capacity = ModelCapacityLimiter(
    max_concurrent_requests=settings.llm_max_concurrent_requests,
    queue_timeout_seconds=settings.llm_queue_timeout_seconds,
)
