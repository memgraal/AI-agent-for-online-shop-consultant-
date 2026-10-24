"""Public question collection and protected FAQ publication endpoints."""

import hmac
import logging
import re
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import delete, desc, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from faq_workbench.batching import should_rebuild_candidates
from faq_workbench.clustering import cluster_questions
from faq_workbench.database import FAQAnalysisState, FAQCandidate, FAQQuestion, FAQQuestionEvent
from faq_workbench.embeddings import embed_texts
from faq_workbench.logging_config import log_event
from faq_workbench.schemas import (
    CandidateResponse,
    PublicFAQItem,
    QuestionSubmission,
    validate_ai_analysis_response,
)
from faq_workbench.settings import settings

router = APIRouter(prefix="/api", tags=["FAQ"])
logger = logging.getLogger("faq_workbench.api")


async def require_faq_admin(x_faq_admin_key: str | None = Header(default=None)) -> None:
    if not settings.admin_api_key:
        raise HTTPException(status_code=503, detail="Административный API FAQ не настроен.")
    if x_faq_admin_key is None or not hmac.compare_digest(x_faq_admin_key, settings.admin_api_key):
        raise HTTPException(status_code=401, detail="Требуется доступ администратора FAQ.")


def _redact(text: str) -> str:
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-zА-Яа-я]{2,}\b", "[email]", text)
    text = re.sub(r"(?<!\w)(?:\+?\d[\d\s().-]{8,}\d)(?!\w)", "[телефон]", text)
    return text.strip()


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


async def _session(request: Request):
    async with request.app.state.session_factory() as session:
        yield session


@router.post("/questions", status_code=202)
async def submit_question(
    payload: QuestionSubmission,
    session: AsyncSession = Depends(_session),
) -> dict[str, str | bool]:
    """Store a submitted chat question; event_id makes client retries idempotent."""
    text = _redact(payload.question)
    normalized = _normalized(text)
    if len(normalized) < 3:
        raise HTTPException(status_code=422, detail="После обработки вопрос слишком короткий.")

    now = datetime.now(timezone.utc)
    question_insert = insert(FAQQuestion).values(
        normalized_text=normalized,
        question_text=text,
        occurrence_count=0,
        first_seen_at=now,
        last_seen_at=now,
    ).on_conflict_do_nothing(index_elements=[FAQQuestion.normalized_text]).returning(FAQQuestion.id)
    question_id = (await session.execute(question_insert)).scalar_one_or_none()
    if question_id is None:
        question_id = (await session.execute(
            select(FAQQuestion.id).where(FAQQuestion.normalized_text == normalized).with_for_update()
        )).scalar_one()

    event_id = str(payload.event_id or uuid4())
    event_insert = insert(FAQQuestionEvent).values(
        event_id=event_id,
        question_id=question_id,
        conversation_id=payload.conversation_id,
    ).on_conflict_do_nothing(index_elements=[FAQQuestionEvent.event_id]).returning(FAQQuestionEvent.event_id)
    inserted_event = (await session.execute(event_insert)).scalar_one_or_none()
    if not inserted_event:
        existing_question_id = (await session.execute(
            select(FAQQuestionEvent.question_id).where(
                FAQQuestionEvent.event_id == event_id
            )
        )).scalar_one_or_none()
        if existing_question_id != question_id:
            # Reusing an idempotency key for a different question must not leave
            # an uncounted aggregate row behind or acknowledge the wrong event.
            await session.rollback()
            raise HTTPException(
                status_code=409,
                detail="event_id уже использован для другого вопроса.",
            )
    if inserted_event:
        await session.execute(
            FAQQuestion.__table__.update()
            .where(FAQQuestion.id == question_id)
            .values(occurrence_count=FAQQuestion.occurrence_count + 1, last_seen_at=now)
        )
        # Keep existing FAQ priority ordering current without rerunning the model
        # when a known question repeats.
        await session.execute(
            update(FAQCandidate)
            .where(FAQCandidate.source_question_ids.contains([question_id]))
            .values(frequency=FAQCandidate.frequency + 1)
        )
    await session.commit()
    log_event(logger, logging.INFO, "faq.question_received", duplicate_event=not bool(inserted_event))
    return {"accepted": True, "event_id": event_id}


@router.get("/faq")
async def get_published_faq(session: AsyncSession = Depends(_session)) -> dict[str, list[PublicFAQItem]]:
    result = await session.execute(
        select(FAQCandidate).where(FAQCandidate.is_published.is_(True))
        .order_by(desc(FAQCandidate.frequency), FAQCandidate.id)
    )
    candidates = list(result.scalars())
    return {"items": [
        PublicFAQItem(id=row.id, category=row.category, question=row.normalized_question)
        for row in candidates
    ]}


@router.post("/admin/faq/rebuild", dependencies=[Depends(require_faq_admin)])
async def rebuild_candidates(
    request: Request,
    limit: int = 100,
    force: bool = False,
    session: AsyncSession = Depends(_session),
) -> dict[str, object]:
    """Thresholded batch analysis; chat submissions never call an AI model."""
    limit = max(1, min(limit, 100))
    now = datetime.now(timezone.utc)
    total_events = await session.scalar(select(func.count()).select_from(FAQQuestionEvent)) or 0
    total_questions = await session.scalar(select(func.count()).select_from(FAQQuestion)) or 0

    # Serialize scheduled/manual rebuilds without holding a DB connection while
    # embedding or waiting for the remote LLM. A lease recovers after process death.
    await session.execute(
        insert(FAQAnalysisState).values(id=1).on_conflict_do_nothing(index_elements=[FAQAnalysisState.id])
    )
    state = (await session.execute(
        select(FAQAnalysisState).where(FAQAnalysisState.id == 1).with_for_update()
    )).scalar_one()
    if state.lock_until and state.lock_until > now:
        await session.rollback()
        return {"status": "in_progress", "total_events": total_events}

    new_questions = max(0, total_questions - state.last_analyzed_question_count)
    if not should_rebuild_candidates(
        force=force,
        new_questions=new_questions,
        last_analyzed_at=state.last_analyzed_at,
        now=now,
        min_new_questions=settings.analysis_min_new_questions,
        min_interval_hours=settings.analysis_min_interval_hours,
    ):
        await session.rollback()
        return {
            "status": "skipped",
            "reason": "below_threshold_or_interval",
            "new_questions": new_questions,
            "required_new_questions": settings.analysis_min_new_questions,
        }

    state.lock_until = now + timedelta(minutes=settings.analysis_lock_minutes)
    await session.commit()

    try:
        result = await session.execute(
            select(FAQQuestion).where(FAQQuestion.occurrence_count > 0)
            .order_by(desc(FAQQuestion.occurrence_count), desc(FAQQuestion.last_seen_at))
            .limit(limit)
        )
        source_questions = list(result.scalars())
        if not source_questions:
            raise HTTPException(status_code=409, detail="Пока нет накопленных вопросов для анализа.")

        missing_embeddings = [
            question for question in source_questions
            if question.embedding is None or question.embedding_model != settings.embedding_model
        ]
        if missing_embeddings:
            vectors = await run_in_threadpool(
                embed_texts,
                settings.embedding_model,
                [question.question_text for question in missing_embeddings],
            )
            for question, vector in zip(missing_embeddings, vectors, strict=True):
                question.embedding = vector
                question.embedding_model = settings.embedding_model
            await session.commit()

        groups = cluster_questions(
            source_questions,
            similarity_threshold=settings.embedding_similarity_threshold,
        )
        # Keep the model context bounded even if most input questions are unrelated.
        groups.sort(key=lambda group: group.frequency, reverse=True)
        groups = groups[:100]
        if not groups:
            raise HTTPException(status_code=409, detail="Не удалось сформировать группы вопросов.")
        group_sources = {
            index: [
                {"id": question.id, "question": question.question_text, "frequency": question.occurrence_count}
                for question in group.members
            ]
            for index, group in enumerate(groups, start=1)
        }
        analysis_questions = [group.analysis_input() for group in groups]
        # Do not keep a pooled PostgreSQL connection open while calling the LLM API.
        await session.rollback()

        client: httpx.AsyncClient = request.app.state.ai_client
        if not settings.ai_admin_api_key:
            raise HTTPException(status_code=503, detail="Не настроен доступ к API основной AI-системы.")
        response = await client.post(
            "/api/admin/faq/analyze",
            headers={"X-Admin-Key": settings.ai_admin_api_key},
            json={"questions": analysis_questions},
        )
        if response.is_error:
            raise HTTPException(status_code=502, detail="AI-система отклонила анализ FAQ.")

        try:
            analysis = validate_ai_analysis_response(
                response.json(),
                expected_question_count=len(groups),
            )
        except (ValueError, httpx.DecodingError) as exc:
            raise HTTPException(
                status_code=502,
                detail="AI-система вернула некорректный анализ FAQ.",
            ) from exc

        candidates = analysis.clusters
        await session.execute(delete(FAQCandidate).where(FAQCandidate.is_published.is_(False)))
        saved: list[CandidateResponse] = []
        for item in candidates:
            source_data = [
                {"question": source["question"], "frequency": source["frequency"]}
                for source_index in item.source_indexes
                for source in group_sources.get(source_index, [])
            ]
            source_ids = [
                str(source["id"])
                for source_index in item.source_indexes
                for source in group_sources.get(source_index, [])
            ]
            candidate = FAQCandidate(
                category=item.category,
                normalized_question=item.normalized_question,
                frequency=item.frequency,
                faq_score=item.faq_score,
                priority=item.priority,
                rationale=item.rationale,
                source_indexes=item.source_indexes,
                source_questions=source_data,
                source_question_ids=source_ids,
                is_published=False,
            )
            session.add(candidate)
            await session.flush()
            saved.append(CandidateResponse(
                id=candidate.id,
                category=candidate.category,
                question=candidate.normalized_question,
                frequency=candidate.frequency,
                score=candidate.faq_score,
                priority=candidate.priority,
                rationale=candidate.rationale,
                source_questions=source_data,
            ))

        state = await session.get(FAQAnalysisState, 1)
        assert state is not None
        state.last_analyzed_question_count = total_questions
        state.last_analyzed_at = datetime.now(timezone.utc)
        state.lock_until = None
        await session.commit()
        log_event(
            logger,
            logging.INFO,
            "faq.candidates_rebuilt",
            question_count=len(source_questions),
            candidate_count=len(saved),
        )
        return {
            "status": "completed",
            "total_events": total_events,
            "new_questions": new_questions,
            "question_count": len(source_questions),
            "semantic_group_count": len(groups),
            "items": saved,
        }
    except httpx.TimeoutException as exc:
        await _release_analysis_lease(session)
        raise HTTPException(status_code=504, detail="AI-система не успела завершить анализ.") from exc
    except httpx.HTTPError as exc:
        await _release_analysis_lease(session)
        raise HTTPException(status_code=502, detail="Не удалось связаться с AI-системой.") from exc
    except Exception:
        await _release_analysis_lease(session)
        raise


async def _release_analysis_lease(session: AsyncSession) -> None:
    await session.rollback()
    state = await session.get(FAQAnalysisState, 1)
    if state is not None:
        state.lock_until = None
        await session.commit()


@router.get("/admin/faq/candidates", dependencies=[Depends(require_faq_admin)])
async def get_candidates(session: AsyncSession = Depends(_session)) -> dict[str, list[CandidateResponse]]:
    result = await session.execute(select(FAQCandidate).order_by(desc(FAQCandidate.frequency), FAQCandidate.id))
    return {"items": [CandidateResponse(
        id=row.id,
        category=row.category,
        question=row.normalized_question,
        frequency=row.frequency,
        score=row.faq_score,
        priority=row.priority,
        rationale=row.rationale,
        source_questions=row.source_questions,
        published=row.is_published,
    ) for row in result.scalars()]}


@router.get("/admin/faq/questions", dependencies=[Depends(require_faq_admin)])
async def get_question_frequencies(
    limit: int = 100,
    session: AsyncSession = Depends(_session),
) -> dict[str, object]:
    limit = max(1, min(limit, 500))
    total_events = await session.scalar(select(func.count()).select_from(FAQQuestionEvent)) or 0
    result = await session.execute(
        select(FAQQuestion.question_text, FAQQuestion.occurrence_count)
        .order_by(desc(FAQQuestion.occurrence_count), desc(FAQQuestion.last_seen_at))
        .limit(limit)
    )
    return {
        "total_events": total_events,
        "items": [
            {"question": question, "frequency": frequency}
            for question, frequency in result.all()
        ],
    }


@router.post("/admin/faq/candidates/{candidate_id}/publish", dependencies=[Depends(require_faq_admin)])
async def publish_candidate(candidate_id: int, session: AsyncSession = Depends(_session)) -> dict[str, bool]:
    candidate = await session.get(FAQCandidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Кандидат FAQ не найден.")
    candidate.is_published = True
    await session.commit()
    log_event(logger, logging.INFO, "faq.candidate_published", candidate_id=candidate_id)
    return {"published": True}


@router.delete("/admin/faq/candidates/{candidate_id}", dependencies=[Depends(require_faq_admin)])
async def unpublish_candidate(candidate_id: int, session: AsyncSession = Depends(_session)) -> dict[str, bool]:
    candidate = await session.get(FAQCandidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Кандидат FAQ не найден.")
    candidate.is_published = False
    await session.commit()
    log_event(logger, logging.INFO, "faq.candidate_unpublished", candidate_id=candidate_id)
    return {"published": False}
