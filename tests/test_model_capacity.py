import asyncio

import pytest

from app.services.model_capacity import ModelCapacityLimiter, ModelOverloadedError


async def test_model_capacity_bounds_concurrent_calls():
    limiter = ModelCapacityLimiter(max_concurrent_requests=2, queue_timeout_seconds=1)
    active = 0
    peak = 0

    async def request() -> None:
        nonlocal active, peak
        async with limiter.slot():
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1

    await asyncio.gather(*(request() for _ in range(8)))

    assert peak == 2


async def test_model_capacity_fails_fast_when_queue_wait_expires():
    limiter = ModelCapacityLimiter(max_concurrent_requests=1, queue_timeout_seconds=0.01)
    entered = asyncio.Event()

    async def held_request() -> None:
        async with limiter.slot():
            entered.set()
            await asyncio.sleep(0.05)

    first = asyncio.create_task(held_request())
    await entered.wait()
    try:
        with pytest.raises(ModelOverloadedError):
            async with limiter.slot():
                pass
    finally:
        await first
