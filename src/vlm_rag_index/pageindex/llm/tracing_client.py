from typing import Any, TypeVar

from pydantic import BaseModel

from vlm_rag_index.pageindex.llm.protocol import LLMClient, Message
from vlm_rag_index.pageindex.observability import observe, record_llm_input

T = TypeVar("T", bound=BaseModel)


class TracingLLMClient:
    """Decorator wrapping an inner `LLMClient` and routing every call through `@observe`.

    When `settings.tracing_enabled` is false the shim's `observe` is a passthrough, so
    this wrapper imposes no overhead. Raw `messages` are not auto-captured (they can hold
    multi-MB inline images); each method records a trace-safe input via `record_llm_input`
    instead, with page images offloaded as Langfuse media.
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner

    @observe(capture_input=False)
    def complete(self, messages: list[Message], **kwargs: Any) -> str:
        record_llm_input(messages)
        return self._inner.complete(messages, **kwargs)

    @observe(capture_input=False)
    async def acomplete(self, messages: list[Message], **kwargs: Any) -> str:
        record_llm_input(messages)
        return await self._inner.acomplete(messages, **kwargs)

    @observe(capture_input=False)
    def complete_structured(self, messages: list[Message], schema: type[T], **kwargs: Any) -> T:
        record_llm_input(messages)
        return self._inner.complete_structured(messages, schema, **kwargs)

    @observe(capture_input=False)
    async def acomplete_structured(self, messages: list[Message], schema: type[T], **kwargs: Any) -> T:
        record_llm_input(messages)
        return await self._inner.acomplete_structured(messages, schema, **kwargs)

    def count_tokens(self, text: str) -> int:
        return self._inner.count_tokens(text)
