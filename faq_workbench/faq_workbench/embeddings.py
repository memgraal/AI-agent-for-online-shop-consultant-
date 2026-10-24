"""Lazy local multilingual embeddings; no per-message LLM/API token usage."""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastembed import TextEmbedding

_model_lock = Lock()
_model: TextEmbedding | None = None
_model_name: str | None = None


def _load_model(model_name: str) -> TextEmbedding:
    # FastEmbed may download the model on the first batch run; protect concurrent
    # initialization because several admin/scheduler requests may overlap.
    global _model, _model_name
    with _model_lock:
        if _model is not None and _model_name == model_name:
            return _model
        from fastembed import TextEmbedding

        _model = TextEmbedding(model_name=model_name)
        _model_name = model_name
        return _model


def embed_texts(model_name: str, texts: list[str]) -> list[list[float]]:
    model = _load_model(model_name)
    return [vector.tolist() for vector in model.embed(texts, batch_size=64)]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)
