from app.agents.common import build_tool_agent, model_node
from app.agents.tool_selection import select_customer_tools
from app.prompts.customer import CUSTOMER_SYSTEM_PROMPT
from app.providers.chat_model import create_chat_model
from app.services.content_controls import content_controls
from app.tools.customer import CUSTOMER_TOOLS


def build_customer_agent():
    llm = create_chat_model()
    return build_tool_agent(
        model_node(
            llm,
            tools=CUSTOMER_TOOLS,
            tool_selector=select_customer_tools,
            topic_guard=content_controls.disabled_topic_response,
            prompt=lambda: f"{CUSTOMER_SYSTEM_PROMPT}\n\n{content_controls.prompt_overlay()}",
        ),
        CUSTOMER_TOOLS,
    )
