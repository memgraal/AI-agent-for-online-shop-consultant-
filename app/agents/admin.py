from app.agents.common import build_tool_agent, model_node
from app.prompts.admin import ADMIN_SYSTEM_PROMPT
from app.providers.chat_model import create_chat_model
from app.tools.admin import ADMIN_TOOLS


def build_admin_agent():
    llm = create_chat_model().bind_tools(ADMIN_TOOLS)
    return build_tool_agent(model_node(llm, ADMIN_SYSTEM_PROMPT), ADMIN_TOOLS)
