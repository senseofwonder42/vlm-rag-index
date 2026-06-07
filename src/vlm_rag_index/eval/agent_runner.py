"""Drive the ReAct agent on one question and recover the pages it chose.

The agent reads pages by calling the `answer_from_pages(doc_name, page_indices,
question)` tool. Those calls are preserved in the returned message list as
`AIMessage.tool_calls`, so page-selection can be evaluated by reading them back
out — no need to instrument the tool itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from vlm_rag_index.agent.tracing import trace_run


@dataclass
class AgentRun:
    """The outcome of running the agent on one question.

    Attributes:
        final_answer: The agent's final free-text answer (empty on error).
        calls: One `(doc_name, page_indices)` per `answer_from_pages` tool call,
            in invocation order.
        error: Error string if the run raised, else None.
    """

    final_answer: str = ""
    calls: list[tuple[str, list[int]]] = field(default_factory=list)
    error: str | None = None


def _extract_calls(messages: list) -> list[tuple[str, list[int]]]:
    calls: list[tuple[str, list[int]]] = []
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            if call.get("name") != "answer_from_pages":
                continue
            args = call.get("args", {})
            doc_name = args.get("doc_name", "")
            pages = args.get("page_indices", []) or []
            calls.append((doc_name, [int(p) for p in pages]))
    return calls


def _final_text(messages: list) -> str:
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


async def run_query(
    graph: CompiledStateGraph,
    question: str,
    *,
    recursion_limit: int = 50,
) -> AgentRun:
    """Run `graph` on `question` and return its answer plus page choices.

    Args:
        graph: The compiled ReAct agent (from `agent.graph.build_graph`).
        question: The user question to ask.
        recursion_limit: Max agent steps before LangGraph aborts the run.

    Returns:
        An `AgentRun`; failures are caught and reported via its `error` field so
        a single bad query never aborts a batch.
    """
    try:
        with trace_run("ask", input=question) as callbacks:
            result = await graph.ainvoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"recursion_limit": recursion_limit, "callbacks": callbacks},
            )
    except Exception as exc:  # noqa: BLE001 - one row's failure must not kill the batch
        logger.warning("query failed: {}", exc)
        return AgentRun(error=str(exc))

    messages = result.get("messages", [])
    return AgentRun(
        final_answer=_final_text(messages),
        calls=_extract_calls(messages),
    )
