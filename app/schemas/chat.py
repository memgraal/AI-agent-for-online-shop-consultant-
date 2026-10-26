"""Chat request and response contracts shared by HTTP and WebSocket APIs."""

from pydantic import BaseModel, ConfigDict, Field

from app.enums import ChatDetail, ChatRole


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: ChatRole
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=12000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=40)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=64)
    detail: ChatDetail = ChatDetail.QUICK


class WebSocketChatRequest(ChatRequest):
    """One customer chat operation sent over the persistent WebSocket."""

    request_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9._-]+$",
    )
    stream: bool = False


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
