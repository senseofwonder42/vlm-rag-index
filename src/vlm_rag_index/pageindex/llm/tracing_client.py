from typing import Any, TypeVar

from pydantic import BaseModel

from vlm_rag_index.pageindex.llm.protocol import LLMClient, Message
from vlm_rag_index.pageindex.observability import observe

T = TypeVar("T", bound=BaseModel)


class TracingLLMClient:
    """Decorator wrapping an inner `LLMClient` and routing every call through `@observe`.

    When `settings.tracing_enabled` is false the shim's `observe` is a passthrough, so
    this wrapper imposes no overhead. Per-call payload extraction (model id, token usage)
    is wired in phase 1 alongside the real underlying client.
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner

    @observe()
    def complete(self, messages: list[Message], **kwargs: Any) -> str:
        return self._inner.complete(messages, **kwargs)

    @observe()
    async def acomplete(self, messages: list[Message], **kwargs: Any) -> str:
        return await self._inner.acomplete(messages, **kwargs)

    @observe()
    def complete_structured(self, messages: list[Message], schema: type[T], **kwargs: Any) -> T:
        return self._inner.complete_structured(messages, schema, **kwargs)

    @observe()
    async def acomplete_structured(self, messages: list[Message], schema: type[T], **kwargs: Any) -> T:
        return await self._inner.acomplete_structured(messages, schema, **kwargs)

    def count_tokens(self, text: str) -> int:
        return self._inner.count_tokens(text)
