"""The LangGraph ReAct agent exposed to `langgraph dev` and agent-chat-ui.

`build_graph()` wires `ChatOpenAI` (the agent's reasoning model, pointed at the
same OpenAI-compatible endpoint the pipeline uses) to the tool surface in
`agent.tools`. The compiled `graph` is what `langgraph.json` references.

LangChain is contained to this layer; the only `ChatOpenAI` in the project lives
here.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from vlm_rag_index.agent.tools import TOOLS
from vlm_rag_index.core.config import settings
from vlm_rag_index.pageindex.prompts import render


def build_graph() -> CompiledStateGraph:
    """Compile the ReAct agent graph from settings, tools, and the system prompt."""
    model = ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        temperature=settings.llm_temperature,
    )
    return create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=render("agent_system.j2"),
    )


graph = build_graph()
