"""Langfuse callback wiring for the LangGraph agent.

Returns an empty list when tracing is disabled (the default) so the agent runs
with zero tracing overhead and no missing-credentials warnings. When enabled,
it lazily builds a Langfuse LangChain `CallbackHandler`; if the integration is
unavailable it logs a warning and degrades to no tracing rather than failing the
request.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from vlm_rag_index.core.config import settings


def langchain_callbacks() -> list[Any]:
    """Callback handlers to pass via `RunnableConfig(callbacks=...)` per request."""
    if not settings.tracing_enabled:
        return []
    try:
        from langfuse.langchain import CallbackHandler
    except Exception as exc:  # integration not installed / unavailable
        logger.warning("tracing enabled but Langfuse LangChain handler unavailable: {}", exc)
        return []
    return [CallbackHandler()]
