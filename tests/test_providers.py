import pytest

from app.config import Settings
from app.providers.chat_model import (
    ROUTERAI_BASE_URL,
    YANDEX_AI_STUDIO_BASE_URL,
    create_chat_model,
)
from app.providers.jev_client import create_jev_client


def test_routerai_provider_uses_routerai_endpoint_and_bearer_key():
    model = create_chat_model(Settings(
        llm_provider="routerai",
        llm_api_key="routerai-secret",
        llm_model="openai/gpt-4o-mini",
        llm_base_url="",
    ))

    assert model.model_name == "openai/gpt-4o-mini"
    assert model.openai_api_base == ROUTERAI_BASE_URL
    assert model.openai_api_key.get_secret_value() == "routerai-secret"


def test_routerai_provider_allows_custom_base_url():
    model = create_chat_model(Settings(
        llm_provider="routerai",
        llm_api_key="routerai-secret",
        llm_model="openai/gpt-4o-mini",
        llm_base_url="https://routerai.ru/custom/v1",
    ))

    assert model.openai_api_base == "https://routerai.ru/custom/v1"


def test_jev_client_uses_independent_typesafe_settings():
    config = Settings(
        llm_provider="yandex",
        llm_api_key="chat-secret",
        llm_model="gpt://folder-123/yandexgpt/latest",
        llm_base_url=YANDEX_AI_STUDIO_BASE_URL,
        jev_llm_api_key="jev-secret",
        jev_llm_model="typesafe/jev-1.14",
        jev_llm_base_url="https://routerai.ru/api",
    )

    chat_model = create_chat_model(config)
    jev_client = create_jev_client(config)

    assert chat_model.openai_api_base == YANDEX_AI_STUDIO_BASE_URL
    assert chat_model.openai_api_key.get_secret_value() == "chat-secret"
    assert jev_client._config.base_url == "https://routerai.ru/api"
    assert jev_client._config.default_model == "typesafe/jev-1.14"
    assert jev_client._config.api_key == "jev-secret"


def test_yandex_provider_uses_ai_studio_endpoint_and_openai_compatible_key():
    model = create_chat_model(Settings(
        llm_provider="yandex",
        llm_api_key="yc-secret",
        llm_model="gpt://folder-123/yandexgpt/latest",
        llm_base_url="",
    ))

    assert model.model_name == "gpt://folder-123/yandexgpt/latest"
    assert model.openai_api_base == YANDEX_AI_STUDIO_BASE_URL
    assert model.openai_api_key.get_secret_value() == "yc-secret"
    assert model.reasoning_effort is None


def test_yandex_qwen_36_disables_reasoning_for_short_agent_responses():
    model = create_chat_model(Settings(
        llm_provider="yandex",
        llm_api_key="yc-secret",
        llm_model="gpt://folder-123/qwen3.6-35b-a3b/latest",
    ))

    assert model.reasoning_effort == "none"


def test_yandex_provider_requires_resource_model_uri():
    with pytest.raises(RuntimeError, match="gpt://"):
        create_chat_model(Settings(
            llm_provider="yandex",
            llm_api_key="yc-secret",
            llm_model="yandexgpt-5-pro",
        ))


def test_openai_compatible_provider_requires_base_url():
    with pytest.raises(RuntimeError, match="LLM_BASE_URL"):
        create_chat_model(Settings(
            llm_provider="openai-compatible",
            llm_model="local-model",
        llm_base_url="",
        ))


def test_openai_compatible_provider_can_use_local_endpoint_without_secret():
    model = create_chat_model(Settings(
        llm_provider="openai-compatible",
        llm_model="qwen-local",
        llm_base_url="http://localhost:11434/v1",
    ))

    assert model.model_name == "qwen-local"
    assert model.openai_api_base == "http://localhost:11434/v1"
