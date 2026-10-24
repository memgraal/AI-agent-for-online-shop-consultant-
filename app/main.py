import logging
import re
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.admin import router as admin_router
from app.api.chat import router as chat_router
from app.api.faq import router as faq_router
from app.api.orders import router as orders_router
from app.api.operator import router as operator_router
from app.config import settings
from app.services.logging import close_logging, configure_logging, log_event, request_id_context
from app.services.rate_limit import rate_limiter

configure_logging()

logger = logging.getLogger("app.http")


@asynccontextmanager
async def lifespan(_: FastAPI):
    log_event(logger, logging.INFO, "application.started", database="not_configured")
    try:
        yield
    finally:
        log_event(logger, logging.INFO, "application.stopping")
        close_logging()


app = FastAPI(title="SSS AI Assistant", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Admin-Key", "X-Request-ID"],
)


@app.middleware("http")
async def enforce_browser_origin(request: Request, call_next):
    origin = request.headers.get("origin")

    if request.url.path.startswith("/api/chat") and not origin:
        return JSONResponse(status_code=403, content={"detail": "Origin header is required."})
    
    if origin and origin.rstrip("/") not in settings.cors_origins:
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

app.include_router(chat_router)
app.include_router(admin_router)
app.include_router(faq_router)
app.include_router(orders_router)
app.include_router(operator_router)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    supplied_id = request.headers.get("X-Request-ID", "")
    request_id = supplied_id if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", supplied_id) else str(uuid4())
    token = request_id_context.set(request_id)
    started = perf_counter()
    try:
        response = await call_next(request)
        duration_ms = round((perf_counter() - started) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        log_event(logger, logging.INFO, "http.request_completed", method=request.method,
                  path=request.url.path, status_code=response.status_code,
                  duration_ms=duration_ms)
        return response
    except Exception as exc:
        log_event(logger, logging.ERROR, "http.request_failed", method=request.method,
                  path=request.url.path, duration_ms=round((perf_counter() - started) * 1000, 2),
                  error_type=type(exc).__name__)
        raise
    finally:
        request_id_context.reset(token)


@app.get("/health", tags=["system"])
def health():
    log_event(logger, logging.DEBUG, "health.checked")
    return {"status": "ok", "database": "not_configured"}


def _provider_configured(provider: str, model: str, api_key: str, base_url: str | None) -> bool:
    if not model.strip():
        return False
    if provider == "openai-compatible":
        return bool(base_url and base_url.strip())
    return bool(api_key.strip())


@app.get("/ready", tags=["system"])
def readiness():
    """Report whether required AI provider and administrative settings exist."""
    checks = {
        "chat_provider": _provider_configured(
            settings.llm_provider,
            settings.llm_model,
            settings.llm_api_key,
            settings.llm_base_url,
        ),
        "jev_provider": bool(
            settings.jev_llm_api_key.strip()
            and settings.jev_llm_model.strip()
            and settings.jev_llm_base_url.strip()
        ),
        "admin_auth": bool(settings.admin_api_key.strip()),
    }
    ready = all(checks.values())
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "checks": checks,
        },
    )
