"""Contracts for the process-local operator handoff prototype."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OperatorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=3, max_length=2000)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=64)


class OperatorTicket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    status: Literal["queued_local"] = "queued_local"
    reason: str
    conversation_id: str | None
    created_at: datetime
    message: str = "Обращение сохранено во временной очереди сервиса; оператор не подключён."
