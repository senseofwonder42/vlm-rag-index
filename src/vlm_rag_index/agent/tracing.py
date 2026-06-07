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

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from loguru import logger

from vlm_rag_index.core.config import settings

RecordOutput = Callable[[Any], None]


@contextmanager
def trace_run(name: str, *, input: Any = None) -> Iterator[tuple[list[Any], RecordOutput]]:
    """Wrap one agent run in a single Langfuse trace.

    Args:
        name: Name for the root span (and thus the trace).
        input: Trace input to record (e.g. the user question).

    Yields:
        A ``(callbacks, record_output)`` pair. `callbacks` go into
        `RunnableConfig(callbacks=...)`; call `record_output(obj)` to attach the
        run's result (answer, pages read) to the root span. Both degrade to no-ops
        when tracing is disabled or unavailable.
    """

    def _noop(_output: Any) -> None:
        return

    if not settings.tracing_enabled:
        yield [], _noop
        return
    try:
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler
    except Exception as exc:  # integration not installed / unavailable
        logger.warning("tracing enabled but Langfuse unavailable: {}", exc)
        yield [], _noop
        return
    client = get_client()
    try:
        with client.start_as_current_observation(
            name=name, as_type="span", input=input
        ) as span:

            def record_output(output: Any) -> None:
                span.update(output=output)

            yield [CallbackHandler()], record_output
    finally:
        # `ask`/`eval` are short-lived processes; flush so late spans aren't lost.
        client.flush()
