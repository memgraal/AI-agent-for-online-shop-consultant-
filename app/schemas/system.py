"""Typed status responses for service health and readiness endpoints."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok", "error"]
    database: Literal["ok", "not_configured", "not_initialized", "unavailable"]


class ReadinessChecks(BaseModel):
    chat_provider: bool
    jev_provider: bool
    admin_auth: bool


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: ReadinessChecks
