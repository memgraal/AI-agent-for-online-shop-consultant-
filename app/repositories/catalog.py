from app.repositories.base import RepositoryResult, not_configured


class CatalogRepository:
    """Contract for product search and comparison; replace with CMS/1C adapter later."""

    async def search(self, query: str, filters: dict) -> RepositoryResult:
        return not_configured("catalog", "search")

    async def compare(self, product_ids: list[str], attributes: list[str]) -> RepositoryResult:
        return not_configured("catalog", "compare")

    async def find_duplicates(self, products: str) -> RepositoryResult:
        return not_configured("catalog", "find_duplicates")

    async def normalize_attributes(self, values: str) -> RepositoryResult:
        return not_configured("catalog", "normalize_attributes")

    async def generate_seo(self, name: str, manufacturer: str, attributes: str) -> RepositoryResult:
        return not_configured("catalog", "generate_seo")
