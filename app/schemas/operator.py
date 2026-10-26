"""Contracts for the process-local operator handoff prototype."""

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field

from app.enums import OperatorStatus


class OperatorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=3, max_length=2000)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=64)


class OperatorTicket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    status: OperatorStatus = OperatorStatus.QUEUED_LOCAL
    reason: str
    conversation_id: str | None
    created_at: datetime
    message: str = "Обращение сохранено во временной очереди сервиса; оператор не подключён."
