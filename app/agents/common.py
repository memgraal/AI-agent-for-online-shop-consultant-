from collections.abc import Callable, Sequence
import logging
from time import perf_counter

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.agents.state import AssistantState
from app.services.agent_settings import agent_settings
from app.services.logging import log_event
from app.services.model_capacity import model_capacity

logger = logging.getLogger("app.agents")


def build_tool_agent(model_call: Callable, tools: Sequence):
    """Build the async-capable ReAct graph: model -> tools -> model."""
    graph = StateGraph(AssistantState)
    graph.add_node("assistant", model_call)
    # A tool failure must fail the request. Converting exceptions into ToolMessages
    # lets the model invent a fallback result (for example, a product SEO draft
    # after structured output validation failed).
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=False))
    graph.add_node("tool_budget_reached", _tool_budget_reached)
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges(
        "assistant", _route_after_assistant,
        {"tools": "tools", "tool_budget_reached": "tool_budget_reached", END: END},
    )
    graph.add_edge("tools", "assistant")
    graph.add_edge("tool_budget_reached", END)
    return graph.compile()


def _route_after_assistant(state: AssistantState) -> str:
    message = state["messages"][-1]
    if not getattr(message, "tool_calls", None):
        return END
    if state.get("llm_calls", 0) >= agent_settings.get().settings.agent_max_llm_calls:
        return "tool_budget_reached"
    return "tools"


async def _tool_budget_reached(state: AssistantState) -> dict:
    runtime = agent_settings.get().settings
    log_event(logger, logging.WARNING, "agent.llm_call_budget_reached",
              llm_calls=state.get("llm_calls", 0), max_llm_calls=runtime.agent_max_llm_calls)
    return {"messages": [AIMessage(content="Не удалось завершить запрос в пределах лимита обращений к модели. Уточните вопрос или попробуйте ещё раз.")]}


def model_node(
    llm,
    prompt: str | Callable[[], str],
    *,
    tools: Sequence | None = None,
    tool_selector=None,
    topic_guard: Callable[[str], str | None] | None = None,
):
    async def call(state: AssistantState) -> dict:
        runtime = agent_settings.get().settings
        detail_instruction = (
            "Дай краткий ответ." if state["detail"] == "quick"
            else "Дай подробный ответ, структуруй его и укажи источники, если они доступны."
        )
        max_output_tokens = runtime.llm_quick_max_output_tokens
        if state["detail"] == "detailed":
            max_output_tokens = runtime.llm_max_output_tokens
        if topic_guard is not None:
            last_user_message = next(
                (message for message in reversed(state["messages"])
                 if isinstance(message, HumanMessage)),
                None,
            )
            blocked_response = (
                topic_guard(str(last_user_message.content))
                if last_user_message is not None
                else None
            )
            if blocked_response is not None:
                log_event(logger, logging.INFO, "agent.topic_disabled")
                return {"messages": [AIMessage(content=blocked_response)]}
        started = perf_counter()
        request_llm = llm
        selected_tools = None
        if tool_selector is not None:
            selected_tools = tool_selector(state)
            request_llm = llm.bind_tools(selected_tools)
            log_event(logger, logging.INFO, "agent.tools_selected",
                      tool_names=[tool.name for tool in selected_tools],
                      available_tool_count=len(tools or ()),
                      selected_tool_count=len(selected_tools))
        log_event(logger, logging.INFO, "llm.request_started",
                  history_count=len(state["messages"]), detail=state["detail"],
                  llm_provider=runtime.llm_provider, llm_model=runtime.llm_model)
        try:
            async with model_capacity.slot():
                response = await request_llm.ainvoke(
                    [SystemMessage(content=f"{prompt() if callable(prompt) else prompt}\n\n{detail_instruction}")] + state["messages"],
                    max_tokens=max_output_tokens,
                )
            usage = response.usage_metadata or {}
            token_details = usage.get("input_token_details") or {}
            log_event(logger, logging.INFO, "llm.request_completed",
                      duration_ms=round((perf_counter() - started) * 1000, 2),
                      input_tokens=usage.get("input_tokens"),
                      cached_input_tokens=token_details.get("cache_read"),
                      output_tokens=usage.get("output_tokens"),
                      total_tokens=usage.get("total_tokens"))
            return {"messages": [response], "llm_calls": state.get("llm_calls", 0) + 1}
        except Exception as exc:
            log_event(logger, logging.ERROR, "llm.request_failed",
                      duration_ms=round((perf_counter() - started) * 1000, 2),
                      error_type=type(exc).__name__)
            raise

    return call
