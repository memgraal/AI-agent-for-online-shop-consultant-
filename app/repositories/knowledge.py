from dataclasses import dataclass
from collections.abc import Sequence
from collections import OrderedDict
import logging
import re

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import settings
from app.repositories.base import RepositoryResult
from app.services.logging import log_event


logger = logging.getLogger("app.repositories")

_FAQ_SEARCH_SQL = text(
    """
    SELECT
        id,
        category,
        normalized_question,
        answer,
        score.relevance
    FROM faq_candidates
    CROSS JOIN LATERAL (
        SELECT COUNT(*)::integer AS relevance
        FROM (
            SELECT DISTINCT term
            FROM regexp_split_to_table(lower(CAST(:query AS text)), '[^[:alnum:]_-]+') AS term
            WHERE length(term) >= 2
            LIMIT 12
        ) AS terms
        WHERE lower(normalized_question || ' ' || answer) ILIKE '%' || terms.term || '%'
    ) AS score
    WHERE is_published AND answer <> '' AND score.relevance > 0
    ORDER BY score.relevance DESC, frequency DESC, id
    LIMIT 5
    """
)


@dataclass(frozen=True)
class KnowledgeChunk:
    """A temporary indexed passage. Replace this store with a vector/CMS adapter later."""

    document_id: str
    title: str
    text: str
    source_url: str = ""
    product_sku: str = ""
    section: str = ""
    page: int | None = None


class KnowledgeCapacityExceeded(RuntimeError):
    """Raised rather than silently evicting indexed knowledge under load."""


class KnowledgeRepository:
    """Search uploaded passages and reviewed FAQ facts from PostgreSQL."""

    def __init__(
        self,
        *,
        max_documents: int | None = None,
        max_chunks: int | None = None,
        max_indexed_chars: int | None = None,
        faq_session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._max_documents = max_documents if max_documents is not None else settings.knowledge_max_documents
        self._max_chunks = max_chunks if max_chunks is not None else settings.knowledge_max_chunks
        self._max_indexed_chars = (
            max_indexed_chars
            if max_indexed_chars is not None
            else settings.knowledge_max_indexed_chars
        )
        self._documents: OrderedDict[str, list[KnowledgeChunk]] = OrderedDict()
        self._chunk_count = 0
        self._indexed_chars = 0
        self._faq_session_factory = faq_session_factory

    def configure_faq(
        self,
        session_factory: async_sessionmaker[AsyncSession] | None,
    ) -> None:
        self._faq_session_factory = session_factory

    async def add_document(
        self, document_id: str, title: str, text: str, *, source_url: str = "",
        product_sku: str = "", section: str = "", page: int | None = None,
    ) -> RepositoryResult:
        if not text.strip():
            return RepositoryResult(configured=True, data={"added": 0, "reason": "empty"})
        chunks = self._make_chunks(
            document_id, title, [(page, text)], source_url, product_sku, section
        )
        self._replace_document(document_id, chunks)
        return RepositoryResult(configured=True, data={"document_id": document_id, "chunks_added": len(chunks)})

    async def add_document_pages(
        self,
        document_id: str,
        title: str,
        pages: Sequence[tuple[int, str]],
        *,
        source_url: str = "",
        product_sku: str = "",
        section: str = "",
    ) -> RepositoryResult:
        """Replace a document while retaining its extracted page citations."""
        chunks = self._make_chunks(
            document_id, title, pages, source_url, product_sku, section
        )
        if not chunks:
            return RepositoryResult(configured=True, data={"added": 0, "reason": "empty"})
        self._replace_document(document_id, chunks)
        return RepositoryResult(configured=True, data={"document_id": document_id, "chunks_added": len(chunks)})

    def _replace_document(self, document_id: str, chunks: list[KnowledgeChunk]) -> None:
        existing = self._documents.get(document_id, [])
        next_document_count = len(self._documents) + (0 if document_id in self._documents else 1)
        next_chunk_count = self._chunk_count - len(existing) + len(chunks)
        next_indexed_chars = self._indexed_chars - sum(len(chunk.text) for chunk in existing)
        next_indexed_chars += sum(len(chunk.text) for chunk in chunks)
        if (
            next_document_count > self._max_documents
            or next_chunk_count > self._max_chunks
            or next_indexed_chars > self._max_indexed_chars
        ):
            raise KnowledgeCapacityExceeded(
                "Индекс базы знаний достиг лимита; освободите место или подключите постоянное хранилище."
            )
        self._documents[document_id] = chunks
        self._chunk_count = next_chunk_count
        self._indexed_chars = next_indexed_chars

    @staticmethod
    def _make_chunks(
        document_id: str,
        title: str,
        pages: Sequence[tuple[int | None, str]],
        source_url: str,
        product_sku: str,
        section: str,
    ) -> list[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        for page, text in pages:
            parts = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
            for part in parts:
                while len(part) > 1200:
                    cut = part.rfind(" ", 0, 1200)
                    cut = cut if cut > 400 else 1200
                    chunks.append(KnowledgeChunk(
                        document_id, title, part[:cut], source_url,
                        product_sku, section, page,
                    ))
                    part = part[cut:].strip()
                if part:
                    chunks.append(KnowledgeChunk(
                        document_id, title, part, source_url,
                        product_sku, section, page,
                    ))
        return chunks

    async def list_documents(self) -> RepositoryResult:
        docs = [
            {
                "document_id": document_id,
                "title": chunks[0].title,
                "source_url": chunks[0].source_url,
                "product_sku": chunks[0].product_sku,
                "chunks": len(chunks),
            }
            for document_id, chunks in self._documents.items()
        ]
        return RepositoryResult(configured=True, data=docs)


    async def search(self, query: str, product: str = "") -> RepositoryResult:
        terms = set(re.findall(r"[\w-]+", query.casefold()))
        product_term = product.casefold().strip()
        ranked = []
        for chunks in self._documents.values():
            for chunk in chunks:
                product_metadata = f"{chunk.product_sku} {chunk.title}".casefold()
                if product_term and product_term not in product_metadata:
                    continue
                text_terms = set(re.findall(r"[\w-]+", chunk.text.casefold()))
                title_terms = set(re.findall(r"[\w-]+", chunk.title.casefold()))
                score = len(terms & text_terms) + 2 * len(terms & title_terms)
                if score:
                    ranked.append((score, chunk))
        ranked.sort(key=lambda item: item[0], reverse=True)
        results: list[dict[str, object]] = [
            {
                "title": c.title,
                "snippet": c.text[:700],
                "sku": c.product_sku or None,
                "section": c.section or None,
                "page": c.page,
                "url": c.source_url or None,
                "relevance": score,
                "source": "uploaded_document",
            }
            for score, c in ranked[:5]
        ]
        results.extend(await self._search_published_faq(query))
        results.sort(key=lambda item: int(item.get("relevance") or 0), reverse=True)
        mode = "uploaded_documents"
        if self._faq_session_factory is not None:
            mode += "+postgresql_published_faq"
        return RepositoryResult(
            configured=True,
            data={"query": query, "results": results[:5], "mode": mode},
        )

    async def _search_published_faq(self, query: str) -> list[dict[str, object]]:
        if self._faq_session_factory is None or not query.strip():
            return []
        try:
            async with self._faq_session_factory() as session:
                result = await session.execute(_FAQ_SEARCH_SQL, {"query": query[:1_000]})
            return [
                {
                    "title": row["normalized_question"],
                    "snippet": row["answer"],
                    "section": row["category"],
                    "relevance": row["relevance"],
                    "source": "reviewed_published_faq",
                    "source_id": row["id"],
                }
                for row in result.mappings().all()
            ]
        except SQLAlchemyError as exc:
            # FAQ is an optional source and can be migrated independently.
            log_event(
                logger,
                logging.WARNING,
                "repository.faq_search_unavailable",
                error_type=type(exc).__name__,
            )
            return []


# Shared only inside this process. A restart clears all indexed content.
knowledge_repository = KnowledgeRepository()
