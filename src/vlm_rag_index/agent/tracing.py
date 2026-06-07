"""Langfuse tracing for the LangGraph agent: one trace per query.

`trace_run` opens a single root span around a whole agent run and yields the
LangChain callbacks to attach to `graph.ainvoke`. Everything nests under that
root via Langfuse's OTEL context: the agent's reasoning steps (through the
`CallbackHandler`) and the `@observe`-decorated VLM/LLM calls alike — so a query
shows up as one linked trace instead of several orphans.

When tracing is disabled (the default) it is a no-op that yields an empty
callbacks list, so callers stay uniform and pay zero overhead. If the Langfuse
integration is unavailable it logs a warning and degrades to no tracing rather
than failing the request.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from loguru import logger

from vlm_rag_index.core.config import settings


@contextmanager
def trace_run(name: str, *, input: Any = None) -> Iterator[list[Any]]:
    """Wrap one agent run in a single Langfuse trace.

    Args:
        name: Name for the root span (and thus the trace).
        input: Trace input to record (e.g. the user question).

    Yields:
        Callback handlers to pass via `RunnableConfig(callbacks=...)`. Empty when
        tracing is disabled or unavailable.
    """
    if not settings.tracing_enabled:
        yield []
        return
    try:
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler
    except Exception as exc:  # integration not installed / unavailable
        logger.warning("tracing enabled but Langfuse unavailable: {}", exc)
        yield []
        return
    with get_client().start_as_current_observation(name=name, as_type="span", input=input):
        yield [CallbackHandler()]
