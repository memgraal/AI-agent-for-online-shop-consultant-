"""Minimal public request API for the in-memory operator queue prototype."""

from fastapi import APIRouter, HTTPException

from app.schemas.operator import OperatorRequest, OperatorTicket
from app.services.operator_handoff import HandoffQueueFull, operator_handoff_queue

router = APIRouter(prefix="/api/operator", tags=["operator handoff"])


@router.post("/requests", response_model=OperatorTicket, status_code=202)
async def create_operator_request(request: OperatorRequest) -> OperatorTicket:
    try:
        return await operator_handoff_queue.create(request)
    except HandoffQueueFull as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
