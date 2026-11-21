"""Read-only retrieval from the nopCommerce PostgreSQL catalog."""

from collections.abc import Mapping

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.repositories.base import RepositoryResult, not_configured


_PRODUCT_DOCUMENTS_SQL = """
WITH product_facts AS (
    SELECT
        p."Id" AS id,
        CAST(p."Name" AS text) AS name,
        CAST(p."Sku" AS text) AS sku,
        CAST(p."ManufacturerPartNumber" AS text) AS manufacturer_part_number,
        regexp_replace(
            COALESCE(CAST(p."ShortDescription" AS text), ''), '<[^>]+>', ' ', 'g'
        ) AS short_description,
        p."Price" AS price,
        p."OldPrice" AS old_price,
        p."CallForPrice" AS call_for_price,
        p."AvailableForPreOrder" AS available_for_preorder,
        CASE
            WHEN p."UseMultipleWarehouses" THEN COALESCE(inventory.stock_quantity, 0)
            ELSE p."StockQuantity"
        END AS stock_quantity,
        COALESCE(manufacturers.names, '') AS manufacturers,
        COALESCE(categories.names, '') AS categories,
        COALESCE(specifications.facts, '') AS specifications,
        CAST(url."Slug" AS text) AS slug,
        CAST(p."FullDescription" AS text) AS full_description
    FROM public."Product" AS p
    LEFT JOIN LATERAL (
        SELECT string_agg(DISTINCT CAST(m."Name" AS text), ', ') AS names
        FROM public."Product_Manufacturer_Mapping" AS pm
        JOIN public."Manufacturer" AS m ON m."Id" = pm."ManufacturerId"
        WHERE pm."ProductId" = p."Id" AND m."Published" AND NOT m."Deleted"
    ) AS manufacturers ON TRUE
    LEFT JOIN LATERAL (
        SELECT string_agg(DISTINCT CAST(c."Name" AS text), ', ') AS names
        FROM public."Product_Category_Mapping" AS pc
        JOIN public."Category" AS c ON c."Id" = pc."CategoryId"
        WHERE pc."ProductId" = p."Id" AND c."Published" AND NOT c."Deleted"
    ) AS categories ON TRUE
    LEFT JOIN LATERAL (
        SELECT string_agg(
            DISTINCT CAST(sa."Name" AS text) || ': ' ||
            COALESCE(NULLIF(CAST(ps."CustomValue" AS text), ''), CAST(sao."Name" AS text)),
            '; '
        ) AS facts
        FROM public."Product_SpecificationAttribute_Mapping" AS ps
        JOIN public."SpecificationAttributeOption" AS sao
            ON sao."Id" = ps."SpecificationAttributeOptionId"
        JOIN public."SpecificationAttribute" AS sa
            ON sa."Id" = sao."SpecificationAttributeId"
        WHERE ps."ProductId" = p."Id" AND ps."ShowOnProductPage"
    ) AS specifications ON TRUE
    LEFT JOIN LATERAL (
        SELECT SUM(pwi."StockQuantity" - pwi."ReservedQuantity")::integer AS stock_quantity
        FROM public."ProductWarehouseInventory" AS pwi
        WHERE pwi."ProductId" = p."Id"
    ) AS inventory ON TRUE
    LEFT JOIN LATERAL (
        SELECT ur."Slug"
        FROM public."UrlRecord" AS ur
        WHERE ur."EntityName" = 'Product'
          AND ur."EntityId" = p."Id"
          AND ur."IsActive"
        ORDER BY (ur."LanguageId" = 0) DESC, ur."Id" DESC
        LIMIT 1
    ) AS url ON TRUE
    WHERE p."Published" AND NOT p."Deleted" AND p."VisibleIndividually"
), product_documents AS (
    SELECT *, lower(concat_ws(
        ' ', name, sku, manufacturer_part_number, manufacturers, categories,
        regexp_replace(COALESCE(short_description, ''), '<[^>]+>', ' ', 'g'),
        regexp_replace(COALESCE(full_description, ''), '<[^>]+>', ' ', 'g'),
        specifications
    )) AS search_document
    FROM product_facts
)
"""

_SEARCH_SQL = text(
    _PRODUCT_DOCUMENTS_SQL
    + """
SELECT
    pd.id, pd.name, pd.sku, pd.manufacturer_part_number,
    pd.short_description, pd.price, pd.old_price, pd.call_for_price,
    pd.available_for_preorder, pd.stock_quantity, pd.manufacturers,
    pd.categories, pd.specifications, pd.slug, score.relevance
FROM product_documents AS pd
CROSS JOIN LATERAL (
    SELECT COUNT(*)::integer AS relevance
    FROM (
        SELECT DISTINCT term
        FROM regexp_split_to_table(lower(CAST(:query AS text)), '[^[:alnum:]_-]+') AS term
        WHERE length(term) >= 2
        LIMIT 12
    ) AS terms
    WHERE pd.search_document ILIKE '%' || terms.term || '%'
) AS score
WHERE score.relevance > 0
ORDER BY score.relevance DESC, pd.name, pd.id
LIMIT :limit
"""
)

_COMPARE_SQL = text(
    _PRODUCT_DOCUMENTS_SQL
    + """
SELECT
    pd.id, pd.name, pd.sku, pd.manufacturer_part_number,
    pd.short_description, pd.price, pd.old_price, pd.call_for_price,
    pd.available_for_preorder, pd.stock_quantity, pd.manufacturers,
    pd.categories, pd.specifications, pd.slug
FROM product_documents AS pd
WHERE CAST(pd.id AS text) = ANY(CAST(:identifiers AS text[]))
   OR lower(COALESCE(pd.sku, '')) = ANY(CAST(:identifiers AS text[]))
ORDER BY pd.name, pd.id
LIMIT 20
"""
)


class CatalogRepository:
    """Retrieve grounded product facts without allowing model-generated SQL."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._session_factory = session_factory

    def configure(
        self,
        session_factory: async_sessionmaker[AsyncSession] | None,
    ) -> None:
        self._session_factory = session_factory

    async def search(self, query: str, filters: dict[str, object]) -> RepositoryResult:
        if self._session_factory is None:
            return not_configured("catalog", "search")

        retrieval_query = " ".join(
            part.strip()
            for part in (
                query,
                str(filters.get("object_type") or ""),
                str(filters.get("requirements") or ""),
            )
            if part.strip()
        )
        if not retrieval_query:
            return RepositoryResult(configured=True, data={"query": query, "results": []})

        async with self._session_factory() as session:
            result = await session.execute(
                _SEARCH_SQL,
                {"query": retrieval_query[:1_000], "limit": 8},
            )
        products = [self._product(row) for row in result.mappings().all()]
        return RepositoryResult(
            configured=True,
            data={
                "query": query,
                "results": products,
                "source": "postgresql_store_catalog",
            },
        )

    async def compare(
        self,
        product_ids: list[str],
        attributes: list[str],
    ) -> RepositoryResult:
        if self._session_factory is None:
            return not_configured("catalog", "compare")

        identifiers = list(
            dict.fromkeys(value.casefold().strip() for value in product_ids if value.strip())
        )[:20]
        if not identifiers:
            return RepositoryResult(configured=True, data={"products": []})

        async with self._session_factory() as session:
            result = await session.execute(_COMPARE_SQL, {"identifiers": identifiers})
        products = [self._product(row) for row in result.mappings().all()]
        if attributes:
            requested = tuple(value.casefold().strip() for value in attributes if value.strip())
            for product in products:
                facts = str(product.get("specifications") or "")
                product["requested_specifications"] = [
                    fact.strip()
                    for fact in facts.split(";")
                    if any(term in fact.casefold() for term in requested)
                ]
        return RepositoryResult(
            configured=True,
            data={"products": products, "source": "postgresql_store_catalog"},
        )

    @staticmethod
    def _product(row: Mapping[str, object]) -> dict[str, object]:
        stock_quantity = row.get("stock_quantity")
        return {
            "id": row.get("id"),
            "name": row.get("name"),
            "sku": row.get("sku"),
            "manufacturer_part_number": row.get("manufacturer_part_number"),
            "manufacturers": row.get("manufacturers"),
            "categories": row.get("categories"),
            "short_description": row.get("short_description"),
            "price": str(row["price"]) if row.get("price") is not None else None,
            "old_price": str(row["old_price"]) if row.get("old_price") is not None else None,
            "call_for_price": row.get("call_for_price"),
            "stock_quantity": stock_quantity,
            "in_stock": isinstance(stock_quantity, int) and stock_quantity > 0,
            "available_for_preorder": row.get("available_for_preorder"),
            "specifications": row.get("specifications"),
            "slug": row.get("slug"),
            "relevance": row.get("relevance"),
        }

    async def find_duplicates(self, products: str) -> RepositoryResult:
        return not_configured("catalog", "find_duplicates")

    async def normalize_attributes(self, values: str) -> RepositoryResult:
        return not_configured("catalog", "normalize_attributes")

    async def generate_seo(
        self,
        name: str,
        manufacturer: str,
        attributes: str,
    ) -> RepositoryResult:
        return not_configured("catalog", "generate_seo")
