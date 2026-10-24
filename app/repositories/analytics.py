from app.repositories.base import RepositoryResult, not_configured


class AnalyticsRepository:
    """Contract for aggregate analytics over anonymized assistant conversations."""

    async def summarize_questions(self, period: str = "") -> RepositoryResult:
        return not_configured("analytics", "summarize_questions")
