"""Bounded, process-local rate limiter for development and single-worker deployments."""

import asyncio
from collections import deque
from time import monotonic

from app.config import settings


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._requests: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    async def check(self, client_key: str) -> tuple[bool, int]:
        """Return whether a request is allowed and the retry delay in seconds."""
        now = monotonic()
        window = float(settings.api_rate_limit_window_seconds)
        async with self._lock:
            queue = self._requests.get(client_key)
            if queue is None:
                self._prune_stale_clients(now, window)
                if len(self._requests) >= settings.api_rate_limit_max_clients:
                    oldest_request = min(
                        request_time
                        for requests in self._requests.values()
                        for request_time in requests
                    )
                    retry_after = max(1, int(oldest_request + window - now + 0.999))
                    return False, retry_after
                queue = deque()
                self._requests[client_key] = queue
            while queue and queue[0] <= now - window:
                queue.popleft()
            if len(queue) >= settings.api_rate_limit_max_requests:
                return False, max(1, int(queue[0] + window - now + 0.999))
            queue.append(now)
            return True, 0

    async def clear(self) -> None:
        """Clear counters, primarily for isolated tests and application shutdown."""
        async with self._lock:
            self._requests.clear()

    def _prune_stale_clients(self, now: float, window: float) -> None:
        stale = [
            key for key, queue in self._requests.items()
            if not queue or queue[-1] <= now - window
        ]
        for key in stale:
            del self._requests[key]


rate_limiter = InMemoryRateLimiter()
 