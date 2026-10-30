"""Validated, non-secret settings that can be changed for AI agents."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.enums import LLMProvider


class AgentSettingsInput(BaseModel):
    """Editable agent settings; provider API keys are deliberately excluded."""

    model_config = ConfigDict(extra="forbid")

    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model: str = Field(min_length=1, max_length=300)
    llm_base_url: str | None = Field(default=None, max_length=2000)
    llm_timeout_seconds: float = Field(default=20.0, ge=1.0, le=300.0)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_retries: int = Field(default=1, ge=0, le=5)
    llm_quick_max_output_tokens: int = Field(default=400, ge=50, le=4096)
    llm_max_output_tokens: int = Field(default=1200, ge=100, le=8192)
    chat_context_max_messages: int = Field(default=20, ge=2, le=40)
    chat_max_context_tokens: int = Field(default=16_000, ge=256, le=32_000)
    agent_recursion_limit: int = Field(default=12, ge=2, le=50)
    agent_max_llm_calls: int = Field(default=6, ge=1, le=10)
    system_prompts: dict[str, str] = Field(min_length=3, max_length=10)

    @field_validator("llm_model", "llm_base_url")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None

    @field_validator("system_prompts")
    @classmethod
    def validate_system_prompts(cls, prompts: dict[str, str]) -> dict[str, str]:
        required_roles = {"customer", "admin", "faq_analysis"}
        if not required_roles.issubset(prompts):
            missing = ", ".join(sorted(required_roles - prompts.keys()))
            raise ValueError(f"Не заданы системные промпты: {missing}.")
        if any(not role.strip() or len(role) > 30 for role in prompts):
            raise ValueError("Идентификатор роли системного промпта задан некорректно.")
        if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts.values()):
            raise ValueError("Системный промпт не может быть пустым.")
        if any(len(prompt) > 30_000 for prompt in prompts.values()):
            raise ValueError("Системный промпт слишком длинный.")
        return {role.strip(): prompt.strip() for role, prompt in prompts.items()}


class AgentSettingsSnapshot(BaseModel):
    revision: int = Field(ge=1)
    updated_at: datetime
    settings: AgentSettingsInput
