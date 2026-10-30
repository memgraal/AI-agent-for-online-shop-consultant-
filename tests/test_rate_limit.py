import pytest
from collections import deque

from app.config import settings
from app.services import rate_limit as rate_limit_module
from app.services.rate_limit import InMemoryRateLimiter


async def test_rate_limiter_does_not_evict_active_clients_when_client_table_is_full(monkeypatch):
    monkeypatch.setattr(settings, "api_rate_limit_max_clients", 1)
    monkeypatch.setattr(settings, "api_rate_limit_max_requests", 1)
    limiter = InMemoryRateLimiter()

    assert await limiter.check("client-a") == (True, 0)
    assert (await limiter.check("client-b"))[0] is False
    assert (await limiter.check("client-a"))[0] is False
    assert list(limiter._requests) == ["client-a"]


async def test_full_client_table_uses_oldest_active_window(monkeypatch):
    monkeypatch.setattr(settings, "api_rate_limit_max_clients", 2)
    monkeypatch.setattr(settings, "api_rate_limit_window_seconds", 60)
    monkeypatch.setattr(rate_limit_module, "monotonic", lambda: 1_000.0)
    limiter = InMemoryRateLimiter()
    limiter._requests = {
        "client-a": deque([950.0, 960.0, 990.0]),
        "client-b": deque([955.0]),
    }

    allowed, retry_after = await limiter.check("client-c")

    assert allowed is False
    assert retry_after == 10
