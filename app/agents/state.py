from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AssistantState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    detail: Literal["quick", "detailed"]
    llm_calls: int
