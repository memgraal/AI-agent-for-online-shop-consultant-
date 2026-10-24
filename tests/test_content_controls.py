from langchain_core.messages import HumanMessage

from app.agents.common import model_node
from app.schemas.content_controls import ContentControlsInput
from app.services.content_controls import ContentControlsStore


def test_content_controls_are_versioned_and_return_defensive_copies():
    store = ContentControlsStore()
    request = ContentControlsInput.model_validate({
        "tone": "concise",
        "disabled_topics": [{"label": "Пиротехника", "match_terms": ["пиротехника"]}],
        "faqs": [{"question": "Есть инструкция?", "answer": "Загрузите модель устройства."}],
    })

    snapshot = store.update(request)
    snapshot.controls.faqs.clear()

    assert snapshot.revision == 2
    assert len(store.get().controls.faqs) == 1
    assert [item.revision for item in store.history()] == [1, 2]
    assert "Отвечай особенно кратко" in store.prompt_overlay()
    assert "Загрузите модель устройства" in store.prompt_overlay()


def test_disabled_topic_matching_uses_terms_not_substrings():
    store = ContentControlsStore()
    store.update(ContentControlsInput.model_validate({
        "disabled_topics": [{"label": "Пиротехника", "match_terms": ["пиротехника"]}],
    }))

    assert store.disabled_topic_response("Расскажите о пиротехнике") is not None
    assert store.disabled_topic_response("Это не пиротехника, а охранная система") is not None
    assert store.disabled_topic_response("Нужен датчик движения") is None


async def test_topic_guard_returns_without_calling_the_model():
    store = ContentControlsStore()
    store.update(ContentControlsInput.model_validate({
        "disabled_topics": [{"label": "Тема", "match_terms": ["закрытая тема"]}],
        "disabled_topic_message": "Эта тема отключена.",
    }))

    class NeverCalledModel:
        async def ainvoke(self, messages, **kwargs):
            raise AssertionError("model must not be called for a disabled topic")

    node = model_node(
        NeverCalledModel(),
        "base prompt",
        topic_guard=store.disabled_topic_response,
    )

    result = await node({
        "messages": [HumanMessage(content="Расскажи про закрытую тему")],
        "detail": "quick",
        "llm_calls": 0,
    })

    assert result["messages"][0].content == "Эта тема отключена."


async def test_prompt_supplier_is_resolved_for_each_model_call():
    class CapturingModel:
        def __init__(self):
            self.messages = None

        async def ainvoke(self, messages, **kwargs):
            self.messages = messages
            from langchain_core.messages import AIMessage
            return AIMessage(content="ok")

    prompt_value = "first version"
    model = CapturingModel()
    node = model_node(model, lambda: prompt_value)
    state = {"messages": [HumanMessage(content="Привет")], "detail": "quick", "llm_calls": 0}

    await node(state)
    prompt_value = "updated version"
    await node(state)

    assert "updated version" in model.messages[0].content
