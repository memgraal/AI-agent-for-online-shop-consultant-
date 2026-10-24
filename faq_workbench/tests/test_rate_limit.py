import httpx
import pytest

from faq_workbench.main import app
from faq_workbench.rate_limit import rate_limiter
from faq_workbench.settings import settings


@pytest.mark.asyncio
async def test_faq_http_api_has_bounded_rate_limit(monkeypatch):
    monkeypatch.setattr(settings, "api_rate_limit_max_requests", 1)
    await rate_limiter.clear()

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            first = await client.get("/not-a-route")
            second = await client.get("/not-a-route")
    finally:
        await rate_limiter.clear()

    assert first.status_code == 404
    assert second.status_code == 429
    assert int(second.headers["Retry-After"]) >= 1
