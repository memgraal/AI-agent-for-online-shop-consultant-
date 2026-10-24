"""Factory for the TypeSafe JEv decision-model client."""

from typesafe_sdk import AsyncTypeSafeClient

from app.config import Settings, settings


def create_jev_client(config: Settings = settings) -> AsyncTypeSafeClient:
    """Create JEv's async SDK client, independently from the chat LLM client."""
    api_key = config.jev_llm_api_key.strip()
    if not api_key:
        raise RuntimeError("Для JEv необходимо задать JEV_LLM_API_KEY.")

    model = config.jev_llm_model.strip()
    if not model:
        raise RuntimeError("JEV_LLM_MODEL не должен быть пустым.")

    base_url = config.jev_llm_base_url.strip()
    if not base_url:
        raise RuntimeError("Для JEv необходимо задать JEV_LLM_BASE_URL.")

    return AsyncTypeSafeClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=config.llm_timeout_seconds,
        compression=False,
    )
