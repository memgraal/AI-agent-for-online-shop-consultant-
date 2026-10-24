from dataclasses import dataclass
from collections.abc import Sequence
from collections import OrderedDict
import re

from app.config import settings
from app.repositories.base import RepositoryResult


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
    """In-memory RAG prototype; content disappears when the process restarts."""

    def __init__(
        self,
        *,
        max_documents: int | None = None,
        max_chunks: int | None = None,
        max_indexed_chars: int | None = None,
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
        results = [{"title": c.title, "snippet": c.text[:700], "sku": c.product_sku or None,
                    "section": c.section or None, "page": c.page,
                    "url": c.source_url or None, "relevance": score}
                   for score, c in ranked[:5]]
        return RepositoryResult(configured=True, data={"query": query, "results": results,
                                                       "mode": "in_memory_keyword_baseline"})


# Shared only inside this process. A restart clears all indexed content.
knowledge_repository = KnowledgeRepository()
