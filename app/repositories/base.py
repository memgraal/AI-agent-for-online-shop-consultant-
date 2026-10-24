from dataclasses import dataclass
import logging

from app.services.logging import log_event

logger = logging.getLogger("app.repositories")


@dataclass(frozen=True)
class RepositoryResult:
    """Explicit result shape used until a real data source is connected."""

    configured: bool
    data: object | None = None
    message: str = ""


def not_configured(repository: str, operation: str) -> RepositoryResult:
    log_event(logger, logging.WARNING, "repository.not_configured",
              repository=repository, operation=operation, configured=False)
    return RepositoryResult(
        configured=False,
        message="Источник данных не подключён; результат не получен.",
    )
