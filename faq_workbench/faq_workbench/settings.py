from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ai_service_url: str = "http://127.0.0.1:8000"
    ai_admin_api_key: str = ""
    ai_api_timeout_seconds: float = Field(default=90, ge=1, le=300)
    api_rate_limit_max_requests: int = Field(default=120, ge=1, le=100_000)
    api_rate_limit_window_seconds: int = Field(default=60, ge=1, le=3600)
    api_rate_limit_max_clients: int = Field(default=10_000, ge=10, le=1_000_000)
    database_url: str = "postgresql+asyncpg://faq:faq@127.0.0.1:5432/sss_faq"
    admin_api_key: str = ""
    allowed_origins: str = ""
    log_level: str = "INFO"
    analysis_min_new_questions: int = Field(default=50, ge=1)
    analysis_min_interval_hours: int = Field(default=24, ge=1)
    analysis_lock_minutes: int = Field(default=10, ge=2)
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_similarity_threshold: float = Field(default=0.82, ge=0.5, le=0.99)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="FAQ_",
    )

    @property
    def parsed_allowed_origins(self) -> list[str]:
        return [origin.strip().rstrip("/") for origin in self.allowed_origins.split(",") if origin.strip()]


settings = Settings()
