"""Approximate token accounting for dialogue context sent to the model."""

from functools import lru_cache
from typing import Protocol

import tiktoken
from langchain_core.messages import BaseMessage


class _TokenEncoder(Protocol):
    def encode(self, text: str) -> list[int]: ...


@lru_cache(maxsize=16)
def _encoder_for(model: str) -> _TokenEncoder:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Yandex and OpenAI-compatible model names may not be in tiktoken's registry.
        return tiktoken.get_encoding("cl100k_base")


def estimate_message_tokens(messages: list[BaseMessage], model: str) -> int:
    """Estimate message tokens, including a small per-message framing allowance."""
    encoder = _encoder_for(model)
    return 2 + sum(len(encoder.encode(str(message.content))) + 4 for message in messages)
