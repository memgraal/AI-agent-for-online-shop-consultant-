from functools import lru_cache
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.agents.admin import build_admin_agent
from app.api.dependencies import require_admin
from app.api.chat_handlers import invoke_chat
from app.config import settings
from app.repositories.knowledge import KnowledgeCapacityExceeded, knowledge_repository
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.admin_product import (
    ProductClassificationDraft,
    ProductContentDraft,
    ProductContentInput,
)
from app.schemas.admin_catalog import (
    AttributeNormalizationRequest,
    AttributeNormalizationResponse,
    DuplicateScanRequest,
    DuplicateScanResponse,
)
from app.schemas.content_controls import ContentControlsInput, ContentControlsSnapshot
from app.schemas.operator import OperatorTicket
from app.schemas.knowledge import KnowledgeDocumentInput, KnowledgeDocumentStored
from app.services.admin_product import classify_product, generate_product_content
from app.services.admin_catalog import find_duplicate_candidates, normalize_product_attributes
from app.services.content_controls import content_controls
from app.services.document_ingestion import (
    DocumentTooLarge,
    InvalidDocument,
    UnsupportedDocumentType,
    extract_document_pages,
)
from app.services.chat import ChatService
from app.services.logging import log_event
from app.services.model_capacity import ModelOverloadedError
from app.services.operator_handoff import operator_handoff_queue

router = APIRouter(
    prefix="/api/admin/assistant",
    tags=["admin assistant"],
    dependencies=[Depends(require_admin)],
)
logger = logging.getLogger("app.api.admin_chat")


@router.post("/knowledge/documents")
async def add_knowledge_document(document: KnowledgeDocumentInput):
    """Add plain text to the process-local prototype index; no file parser or DB yet."""
    try:
        result = await knowledge_repository.add_document(
            document.document_id, document.title, document.text,
            source_url=document.source_url, product_sku=document.product_sku,
            section=document.section, page=document.page,
        )
    except KnowledgeCapacityExceeded as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "60"}) from exc
    return result.data


@router.get("/knowledge/documents")
async def list_knowledge_documents():
    result = await knowledge_repository.list_documents()
    return {"documents": result.data, "storage": "process_memory"}


@router.get("/operator-requests", response_model=list[OperatorTicket])
async def list_operator_requests(
    limit: int = Query(default=100, ge=1, le=500),
) -> list[OperatorTicket]:
    return await operator_handoff_queue.list_pending(limit=limit)


@router.get("/content", response_model=ContentControlsSnapshot)
async def get_content_controls() -> ContentControlsSnapshot:
    return content_controls.get()


@router.put("/content", response_model=ContentControlsSnapshot)
async def update_content_controls(
    controls: ContentControlsInput,
) -> ContentControlsSnapshot:
    snapshot = content_controls.update(controls)
    log_event(
        logger,
        logging.INFO,
        "admin.content_controls_updated",
        revision=snapshot.revision,
    )
    return snapshot


@router.get("/content/history", response_model=list[ContentControlsSnapshot])
async def get_content_control_history() -> list[ContentControlsSnapshot]:
    return content_controls.history()


@router.post(
    "/knowledge/documents/upload",
    status_code=201,
    response_model=KnowledgeDocumentStored,
)
async def upload_knowledge_document(
    document_id: str = Form(min_length=1, max_length=200),
    title: str = Form(min_length=1, max_length=500),
    file: UploadFile = File(...),
    source_url: str = Form(default="", max_length=2000),
    product_sku: str = Form(default="", max_length=200),
    section: str = Form(default="", max_length=500),
) -> KnowledgeDocumentStored:
    """Extract a supported file into process memory without storing the upload."""
    try:
        content = await file.read(settings.knowledge_max_upload_bytes + 1)
        if len(content) > settings.knowledge_max_upload_bytes:
            raise DocumentTooLarge("Размер файла превышает лимит загрузки.")
        pages = await run_in_threadpool(
            extract_document_pages,
            file.filename or "",
            content,
        )
        if all(page is None for page, _ in pages):
            result = await knowledge_repository.add_document(
                document_id,
                title,
                "\n\n".join(text for _, text in pages),
                source_url=source_url,
                product_sku=product_sku,
                section=section,
            )
        else:
            pdf_pages = [(page, text) for page, text in pages if page is not None]
            result = await knowledge_repository.add_document_pages(
                document_id,
                title,
                pdf_pages,
                source_url=source_url,
                product_sku=product_sku,
                section=section,
            )
        if not isinstance(result.data, dict):
            raise RuntimeError("Knowledge repository returned an invalid result.")
        return KnowledgeDocumentStored.model_validate(
            {**result.data, "storage": "process_memory"}
        )
    except UnsupportedDocumentType as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except DocumentTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidDocument as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except KnowledgeCapacityExceeded as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "60"}) from exc
    finally:
        await file.close()


@router.post("/products/seo-draft", response_model=ProductContentDraft)
async def create_product_seo_draft(request: ProductContentInput) -> ProductContentDraft:
    """Return a validated, fixed-schema product draft without chat reformatting."""
    try:
        return await generate_product_content(
            request.name,
            request.manufacturer,
            request.attributes,
        )
    except ModelOverloadedError as exc:
        raise HTTPException(
            status_code=503,
            detail="AI-сервис перегружен. Повторите запрос позже.",
            headers={"Retry-After": "5"},
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "admin.product_draft_request_failed",
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=502,
            detail="Не удалось сформировать структурированный SEO-черновик.",
        ) from exc


@router.post("/products/classify", response_model=ProductClassificationDraft)
async def classify_admin_product(request: ProductContentInput) -> ProductClassificationDraft:
    """Return JEv's validated product classification as a standalone schema."""
    try:
        return await classify_product(
            request.name,
            request.manufacturer,
            request.attributes,
        )
    except ModelOverloadedError as exc:
        raise HTTPException(
            status_code=503,
            detail="AI-сервис перегружен. Повторите запрос позже.",
            headers={"Retry-After": "5"},
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "admin.product_classification_request_failed",
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=502,
            detail="Не удалось классифицировать товар.",
        ) from exc


@router.post(
    "/catalog/normalize-attributes",
    response_model=AttributeNormalizationResponse,
)
async def normalize_catalog_attributes(
    request: AttributeNormalizationRequest,
) -> AttributeNormalizationResponse:
    try:
        return await normalize_product_attributes(request)
    except ModelOverloadedError as exc:
        raise HTTPException(
            status_code=503,
            detail="AI-сервис перегружен. Повторите запрос позже.",
            headers={"Retry-After": "5"},
        ) from exc
    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "admin.attribute_normalization_request_failed",
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=502,
            detail="Не удалось нормализовать атрибуты.",
        ) from exc


@router.post("/catalog/duplicate-candidates", response_model=DuplicateScanResponse)
async def scan_catalog_duplicates(
    request: DuplicateScanRequest,
) -> DuplicateScanResponse:
    return find_duplicate_candidates(request)


@lru_cache(maxsize=1)
def _agent():
    return build_admin_agent()


@lru_cache(maxsize=1)
def _service() -> ChatService:
    return ChatService(
        scope="admin",
        agent_factory=lambda: _agent(),
        logger=logger,
        failure_message="Не удалось выполнить запрос администратора.",
    )


@router.post("/chat", response_model=ChatResponse)
async def admin_chat(request: ChatRequest) -> ChatResponse:
    return await invoke_chat(_service(), request)
