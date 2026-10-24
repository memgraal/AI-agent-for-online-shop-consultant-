import pytest

from app.config import settings
from app.services.rate_limit import InMemoryRateLimiter


async def test_rate_limiter_does_not_evict_active_clients_when_client_table_is_full(monkeypatch):
    monkeypatch.setattr(settings, "api_rate_limit_max_clients", 1)
    monkeypatch.setattr(settings, "api_rate_limit_max_requests", 1)
    limiter = InMemoryRateLimiter()

    assert await limiter.check("client-a") == (True, 0)
    assert (await limiter.check("client-b"))[0] is False
    assert (await limiter.check("client-a"))[0] is False
    assert list(limiter._requests) == ["client-a"]
