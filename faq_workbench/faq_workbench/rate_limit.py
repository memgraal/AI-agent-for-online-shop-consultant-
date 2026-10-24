"""Bounded process-local HTTP rate limiter for the standalone FAQ service."""

import asyncio
from collections import deque
from time import monotonic

from faq_workbench.settings import settings


class FAQRateLimiter:
    def __init__(self) -> None:
        self._requests: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    async def check(self, client_key: str) -> tuple[bool, int]:
        now = monotonic()
        window = float(settings.api_rate_limit_window_seconds)
        async with self._lock:
            queue = self._requests.get(client_key)
            if queue is None:
                self._prune_stale(now, window)
                if len(self._requests) >= settings.api_rate_limit_max_clients:
                    oldest = min(
                        timestamp
                        for requests in self._requests.values()
                        for timestamp in requests
                    )
                    return False, max(1, int(oldest + window - now + 0.999))
                queue = deque()
                self._requests[client_key] = queue
            while queue and queue[0] <= now - window:
                queue.popleft()
            if len(queue) >= settings.api_rate_limit_max_requests:
                return False, max(1, int(queue[0] + window - now + 0.999))
            queue.append(now)
            return True, 0

    async def clear(self) -> None:
        async with self._lock:
            self._requests.clear()

    def _prune_stale(self, now: float, window: float) -> None:
        stale = [
            client for client, timestamps in self._requests.items()
            if not timestamps or timestamps[-1] <= now - window
        ]
        for client in stale:
            del self._requests[client]


rate_limiter = FAQRateLimiter()
