"""Standalone API service for collecting and publishing FAQ questions."""

from contextlib import asynccontextmanager
import logging
import re
from time import perf_counter
from typing import AsyncIterator
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from faq_workbench.api import router
from faq_workbench.database import create_database
from faq_workbench.logging_config import configure_logging, log_event, request_id_context
from faq_workbench.rate_limit import rate_limiter
from faq_workbench.settings import settings

configure_logging(settings.log_level)
logger = logging.getLogger("faq_workbench.http")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    session_factory, engine = create_database(settings.database_url)
    app.state.session_factory = session_factory
    app.state.engine = engine
    app.state.ai_client = httpx.AsyncClient(
        base_url=settings.ai_service_url.rstrip("/"),
        timeout=settings.ai_api_timeout_seconds,
    )
    try:
        yield
    finally:
        await app.state.ai_client.aclose()
        await engine.dispose()


app = FastAPI(title="SSS FAQ API", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.parsed_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-FAQ-Admin-Key", "X-Request-ID"],
)


@app.middleware("http")
async def enforce_browser_origin(request: Request, call_next):
    origin = request.headers.get("origin")
    if request.url.path in {"/api/questions", "/api/faq"} and not origin:
        return JSONResponse(status_code=403, content={"detail": "Origin header is required."})
    if origin and origin.rstrip("/") not in settings.parsed_allowed_origins:
        return JSONResponse(status_code=403, content={"detail": "Origin is not allowed."})
    return await call_next(request)


@app.middleware("http")
async def enforce_rate_limit(request: Request, call_next):
    if request.url.path in {"/health", "/openapi.json", "/docs", "/redoc"}:
        return await call_next(request)
    client_key = request.client.host if request.client else "unknown"
    allowed, retry_after = await rate_limiter.check(client_key)
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"detail": "Слишком много запросов. Повторите позже."},
            headers={"Retry-After": str(retry_after)},
        )
    return await call_next(request)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    supplied_id = request.headers.get("X-Request-ID", "")
    request_id = supplied_id if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", supplied_id) else str(uuid4())
    token = request_id_context.set(request_id)
    started = perf_counter()
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        log_event(
            logger,
            logging.INFO,
            "http.request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round((perf_counter() - started) * 1000, 2),
        )
        return response
    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "http.request_failed",
            method=request.method,
            path=request.url.path,
            duration_ms=round((perf_counter() - started) * 1000, 2),
            error_type=type(exc).__name__,
        )
        raise
    finally:
        request_id_context.reset(token)


app.include_router(router)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    engine: AsyncEngine = app.state.engine
    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))
    return {"status": "ok", "database": "ok"}
