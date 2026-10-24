import hmac
import logging

from fastapi import Header, HTTPException

from app.config import settings
from app.services.logging import log_event

logger = logging.getLogger("app.security")


def require_admin(x_admin_key: str | None = Header(default=None)) -> None:
    
    if not settings.admin_api_key:
        log_event(logger, logging.ERROR, "admin.auth_not_configured")
        raise HTTPException(status_code=503, detail="Административный API не настроен.")
    
    if x_admin_key is None or not hmac.compare_digest(x_admin_key, settings.admin_api_key):
        log_event(logger, logging.WARNING, "admin.auth_failed")
        raise HTTPException(status_code=401, detail="Требуется доступ администратора.")
    
    log_event(logger, logging.INFO, "admin.auth_succeeded")
