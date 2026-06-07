"""The compiled agent graph wires the context-management middleware."""

from langchain.agents.middleware import ClearToolUsesEdit
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from vlm_rag_index.agent.graph import build_graph
from vlm_rag_index.core.config import settings


def test_build_graph_includes_summarization_middleware():
    graph = build_graph()
    nodes = set(graph.get_graph().nodes)
    assert "SummarizationMiddleware.before_model" in nodes


def test_clear_tool_uses_edit_prunes_oversized_history():
    """With our settings, a history past the trigger has old tool outputs cleared."""
    edit = ClearToolUsesEdit(
        trigger=settings.agent_context_edit_trigger_tokens,
        keep=settings.agent_context_edit_keep,
    )
    big = "x " * settings.agent_context_edit_trigger_tokens  # ~1 token per "x "
    messages = [HumanMessage("q")]
    for i in range(settings.agent_context_edit_keep + 3):
        messages.append(AIMessage("", tool_calls=[{"name": "t", "args": {}, "id": str(i)}]))
        messages.append(ToolMessage(content=big, tool_call_id=str(i)))

    edit.apply(messages, count_tokens=lambda m: sum(len(str(x.content)) for x in m))

    cleared = [m for m in messages if isinstance(m, ToolMessage) and big not in str(m.content)]
    assert cleared, "expected oldest tool outputs to be cleared once over the trigger"
