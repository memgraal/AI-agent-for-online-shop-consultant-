from app.repositories.base import RepositoryResult, not_configured


class OrderRepository:
    """Contract for discounts, order totals, legal details and operator requests."""

    async def validate_company_details(self, details: str) -> RepositoryResult:
        return not_configured("orders", "validate_company_details")

    async def calculate_total(self, items: str, customer_type: str, purchase_total: float | None) -> RepositoryResult:
        return not_configured("orders", "calculate_total")

    async def request_operator(self, reason: str) -> RepositoryResult:
        return not_configured("orders", "request_operator")
