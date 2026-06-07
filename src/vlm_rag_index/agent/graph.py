"""The LangGraph ReAct agent exposed to `langgraph dev` and agent-chat-ui.

`build_graph()` wires `ChatOpenAI` (the agent's reasoning model, pointed at the
same OpenAI-compatible endpoint the pipeline uses) to the tool surface in
`agent.tools`. The compiled `graph` is what `langgraph.json` references.

LangChain is contained to this layer; the only `ChatOpenAI` in the project lives
here.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    SummarizationMiddleware,
)
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from vlm_rag_index.agent.tools import TOOLS
from vlm_rag_index.core.config import settings
from vlm_rag_index.pageindex.prompts import render


def build_graph() -> CompiledStateGraph:
    """Compile the ReAct agent graph from settings, tools, and the system prompt.

    Two context-management middleware keep the reasoning model's prompt bounded as
    the conversation grows (long multi-turn chats, many documents read in one run):

    - `ContextEditingMiddleware` clears the oldest bulky tool outputs (structure
      dumps, search lists) once the history crosses a token threshold, keeping the
      last few intact — the question/answer thread is preserved.
    - `SummarizationMiddleware` is the backstop: it summarizes old turns once even
      the pruned history grows past a larger threshold.
    """
    model = ChatOpenAI(
        model=settings.agent_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        temperature=settings.llm_temperature,
    )
    return create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=render("agent_system.j2"),
        middleware=[
            ContextEditingMiddleware(
                edits=[
                    ClearToolUsesEdit(
                        trigger=settings.agent_context_edit_trigger_tokens,
                        keep=settings.agent_context_edit_keep,
                    )
                ]
            ),
            SummarizationMiddleware(
                model=model,
                trigger=("tokens", settings.agent_summary_trigger_tokens),
                keep=("messages", settings.agent_summary_keep_messages),
            ),
        ],
    )


graph = build_graph()
