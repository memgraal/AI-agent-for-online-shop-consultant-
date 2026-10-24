"""Create a LangChain chat model from provider-neutral environment settings."""

import logging
from time import perf_counter

from langchain_openai import ChatOpenAI

from app.config import Settings, settings
from app.services.logging import log_event

YANDEX_AI_STUDIO_BASE_URL = "https://ai.api.cloud.yandex.net/v1"
ROUTERAI_BASE_URL = "https://routerai.ru/api/v1"
logger = logging.getLogger("app.providers.chat_model")


def create_chat_model(config: Settings = settings) -> ChatOpenAI:
    """Build a ChatOpenAI transport for OpenAI, Yandex AI Studio, or compatible APIs."""
    started = perf_counter()
    provider = config.llm_provider
    model = config.llm_model.strip()
    api_key = config.llm_api_key.strip()

    log_event(logger, logging.INFO, "llm.client_creation_started",
              llm_provider=provider, llm_model=model)

    try:
        if provider != "openai-compatible" and not api_key:
            raise RuntimeError(f"Для провайдера {provider} необходимо задать LLM_API_KEY.")

        if not model:
            raise RuntimeError("LLM_MODEL не должен быть пустым.")

        base_url = config.llm_base_url
        default_headers = None

        if provider == "openai":
            base_url = base_url or None
        elif provider == "yandex":
            if not model.startswith("gpt://"):
                raise RuntimeError(
                    "Для Yandex AI Studio укажите LLM_MODEL в формате "
                    "gpt://<folder_id>/<model_id>/latest."
                )
            base_url = base_url or YANDEX_AI_STUDIO_BASE_URL
            # Yandex's own OpenAI-compatible example passes the service key through
            # the standard OpenAI client api_key parameter (Bearer authorization).
        elif provider == "routerai":
            base_url = base_url or ROUTERAI_BASE_URL
        elif provider == "openai-compatible":
            if not base_url:
                raise RuntimeError("Для openai-compatible необходимо задать LLM_BASE_URL.")
        else:  # Settings validates this, retained for callers constructing settings dynamically.
            raise RuntimeError(f"Неподдерживаемый провайдер: {provider}.")

        client = ChatOpenAI(
            model=model,
            api_key=api_key or "local-endpoint",
            base_url=base_url,
            default_headers=default_headers,
            temperature=0,
            timeout=config.llm_timeout_seconds,
            # Qwen 3.6 spends completion tokens on hidden reasoning by default.
            # For a short customer chat this can exhaust the output budget before
            # any user-visible text is produced. Yandex Chat Completions supports
            # reasoning_effort="none" for reasoning models.
            reasoning_effort=(
                "none" if provider == "yandex" and "qwen3.6" in model.lower() else None
            ),
        )
    except Exception as exc:
        log_event(logger, logging.ERROR, "llm.client_creation_failed",
                  llm_provider=provider, llm_model=model,
                  duration_ms=round((perf_counter() - started) * 1000, 2),
                  error_type=type(exc).__name__)
        raise

    log_event(logger, logging.INFO, "llm.client_created",
              llm_provider=provider, llm_model=model,
              duration_ms=round((perf_counter() - started) * 1000, 2))
    return client


