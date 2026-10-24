"""Backward-compatible schema exports; prefer importing feature modules directly."""

from app.schemas.chat import ChatMessage, ChatRequest, ChatResponse, WebSocketChatRequest
from app.schemas.faq import FAQAnalysisRequest, FAQAnalysisResponse, FAQCluster, FAQQuestionInput
from app.schemas.knowledge import KnowledgeDocumentInput

__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "FAQAnalysisRequest",
    "FAQAnalysisResponse",
    "FAQCluster",
    "FAQQuestionInput",
    "KnowledgeDocumentInput",
    "WebSocketChatRequest",
]
