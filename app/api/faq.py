"""Protected HTTP routes for FAQ analysis."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import require_admin
from app.schemas.faq import FAQAnalysisRequest, FAQAnalysisResponse
from app.services.faq_analysis import analyze_faq_questions
from app.services.logging import log_event
from app.services.model_capacity import ModelOverloadedError

router = APIRouter(
    prefix="/api/admin/faq",
    tags=["FAQ analysis"],
    dependencies=[Depends(require_admin)],
)
logger = logging.getLogger("app.api.faq")


@router.post("/analyze", response_model=FAQAnalysisResponse)
async def analyze_faq_endpoint(
    request: FAQAnalysisRequest,
) -> FAQAnalysisResponse:
    """Analyze supplied customer questions and return FAQ candidates."""
    try:
        return await analyze_faq_questions(request.questions)
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
            "faq.analysis_endpoint_failed",
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=502,
            detail="Не удалось проанализировать вопросы для FAQ.",
        ) from exc
