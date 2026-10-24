from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Chat model settings (LLM_*): used by customer and admin conversations.
    llm_provider: Literal["openai", "yandex", "routerai", "openai-compatible"] = "openai"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None
    llm_timeout_seconds: float = 20.0
    llm_max_concurrent_requests: int = Field(default=64, ge=1, le=10_000)
    llm_queue_timeout_seconds: float = Field(default=5.0, ge=0.1, le=120.0)
    # TypeSafe JEv classifier settings (JEV_LLM_*): independent from chat credentials.
    jev_llm_api_key: str = ""
    jev_llm_model: str = "typesafe/jev-1.14"
    jev_llm_base_url: str = "https://routerai.ru/api"
    llm_max_output_tokens: int = Field(default=1200, ge=100, le=8192)
    llm_quick_max_output_tokens: int = Field(default=400, ge=50, le=4096)
    admin_api_key: str = ""
    agent_recursion_limit: int = Field(default=12, ge=2, le=50)
    agent_max_llm_calls: int = Field(default=6, ge=1, le=10)
    chat_context_max_messages: int = Field(default=20, ge=2, le=40)
    chat_max_context_tokens: int = Field(default=16000, ge=256, le=32000)
    chat_max_sessions: int = Field(default=10000, ge=100, le=1_000_000)
    knowledge_max_upload_bytes: int = Field(default=10_485_760, ge=1024, le=104_857_600)
    knowledge_max_text_chars: int = Field(default=200_000, ge=1000, le=2_000_000)
    knowledge_max_pdf_pages: int = Field(default=500, ge=1, le=10_000)
    knowledge_max_documents: int = Field(default=5000, ge=1, le=1_000_000)
    knowledge_max_chunks: int = Field(default=10_000, ge=1, le=1_000_000)
    knowledge_max_indexed_chars: int = Field(default=10_000_000, ge=1000, le=1_000_000_000)
    operator_queue_max_items: int = Field(default=1000, ge=1, le=100_000)
    operator_queue_ttl_seconds: int = Field(default=86_400, ge=60, le=2_592_000)
    api_rate_limit_max_requests: int = Field(default=120, ge=1, le=100_000)
    api_rate_limit_window_seconds: int = Field(default=60, ge=1, le=3600)
    api_rate_limit_max_clients: int = Field(default=10_000, ge=10, le=1_000_000)
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    allowed_origins: str = ""
    log_level: str = "INFO"
    log_file: str = "logs/sss-ai.log"
    log_format: Literal["json", "text"] = "json"
    log_max_bytes: int = 10_485_760
    log_backup_count: int = 5

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip().rstrip("/") for origin in self.allowed_origins.split(",") if origin.strip()]


settings = Settings()
