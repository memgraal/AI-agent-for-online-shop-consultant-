from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from app.agents.common import build_tool_agent


async def test_graph_awaits_async_model_and_async_tool():
    tool_calls: list[str] = []

    @tool
    async def echo(value: str) -> str:
        """Return the value and record that the async tool ran."""
        tool_calls.append(value)
        return f"echo:{value}"

    async def fake_model(state):
        if any(isinstance(message, ToolMessage) for message in state["messages"]):
            return {"messages": [AIMessage(content="Готово") ]}
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "echo",
                        "args": {"value": "async"},
                        "id": "call-1",
                        "type": "tool_call",
                    }],
                )
            ]
        }

    graph = build_tool_agent(fake_model, [echo])
    result = await graph.ainvoke({
        "messages": [HumanMessage(content="Вызови инструмент")],
        "detail": "quick",
    })

    assert tool_calls == ["async"]
    assert result["messages"][-1].content == "Готово"


async def test_graph_propagates_tool_failure_instead_of_asking_model_to_fallback():
    model_calls = 0

    @tool
    async def fail(value: str) -> str:
        """Fail to simulate an unavailable generation provider."""
        raise ValueError("structured output validation failed")

    async def fake_model(state):
        nonlocal model_calls
        model_calls += 1
        return {
            "messages": [AIMessage(content="", tool_calls=[{
                "name": "fail",
                "args": {"value": "x"},
                "id": "call-fail",
                "type": "tool_call",
            }])]
        }

    graph = build_tool_agent(fake_model, [fail])

    try:
        await graph.ainvoke({
            "messages": [HumanMessage(content="Сгенерируй черновик")],
            "detail": "quick",
        })
    except ValueError as exc:
        assert str(exc) == "structured output validation failed"
    else:
        raise AssertionError("tool error should fail the graph invocation")

    assert model_calls == 1


async def test_graph_stops_tool_loop_at_llm_call_budget(monkeypatch):
    from app.agents.common import model_node
    from app.config import settings

    monkeypatch.setattr(settings, "agent_max_llm_calls", 2)
    tool_calls = []

    @tool
    async def repeat(value: str) -> str:
        """Record a bounded tool call."""
        tool_calls.append(value)
        return "done"

    class RepeatingModel:
        async def ainvoke(self, messages, **kwargs):
            return AIMessage(content="", tool_calls=[{
                "name": "repeat", "args": {"value": "x"}, "id": f"call-{len(tool_calls)}",
                "type": "tool_call",
            }])

    graph = build_tool_agent(model_node(RepeatingModel(), "prompt"), [repeat])
    result = await graph.ainvoke({
        "messages": [HumanMessage(content="Продолжай")], "detail": "quick", "llm_calls": 0,
    })

    assert len(tool_calls) == 1
    assert "лимита обращений" in result["messages"][-1].content
