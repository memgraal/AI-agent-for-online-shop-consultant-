import logging
from collections.abc import Awaitable, Callable
from time import perf_counter

from app.repositories.base import RepositoryResult
from app.services.logging import log_event

logger = logging.getLogger("app.tools")


async def run_tool(
    tool_name: str,
    operation: Callable[[], Awaitable[RepositoryResult]],
) -> str:
    """Log tool lifecycle metadata without logging tool arguments or returned content."""
    started = perf_counter()
    log_event(logger, logging.INFO, "tool.started", tool=tool_name)
    try:
        result = await operation()
        duration_ms = round((perf_counter() - started) * 1000, 2)
        if not result.configured:
            log_event(logger, logging.WARNING, "tool.source_not_configured",
                      tool=tool_name, duration_ms=duration_ms)
            return result.message
        log_event(logger, logging.INFO, "tool.completed", tool=tool_name,
                  configured=True, duration_ms=duration_ms)
        return str(result.data)
    except Exception as exc:
        log_event(logger, logging.ERROR, "tool.failed", tool=tool_name,
                  duration_ms=round((perf_counter() - started) * 1000, 2),
                  error_type=type(exc).__name__)
        raise
