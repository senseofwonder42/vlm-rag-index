from typing import Any, Callable, TypeVar, cast

from vlm_rag_index.core.config import settings
from vlm_rag_index.pageindex.llm.protocol import Message

F = TypeVar("F", bound=Callable[..., Any])


def _trace_safe_messages(
    messages: list[Message], to_media: Callable[[str], Any]
) -> list[dict[str, Any]]:
    """Copy `messages`, keeping text parts and routing image data-URIs through `to_media`."""
    safe: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            safe.append({"role": message.get("role"), "content": content})
            continue
        parts: list[Any] = []
        for part in content or []:
            p = cast("dict[str, Any]", part)
            if p.get("type") == "image_url":
                parts.append({"type": "image", "data": to_media(p["image_url"]["url"])})
            else:
                parts.append(part)
        safe.append({"role": message.get("role"), "content": parts})
    return safe


def record_llm_input(messages: list[Message]) -> None:
    """Attach a trace-safe view of `messages` to the current Langfuse span.

    Text parts are kept verbatim; each inline base64 image is offloaded as Langfuse
    media so the rendered pages stay viewable in the trace without bloating span
    attributes (the multi-MB data URI is replaced by a media reference). No-op when
    tracing is disabled, so callers stay uniform.
    """
    if not settings.tracing_enabled:
        return
    from langfuse import get_client
    from langfuse.media import LangfuseMedia

    safe = _trace_safe_messages(messages, lambda url: LangfuseMedia(base64_data_uri=url))
    get_client().update_current_span(input=safe)


def observe(*dargs: Any, **dkwargs: Any) -> Callable[[F], F]:
    """No-op `@observe` decorator unless `settings.tracing_enabled` is true.

    Importing Langfuse lazily avoids its "missing credentials" warning when tracing
    is off (the default). When enabled, forward to `langfuse.observe`.
    """
    if not settings.tracing_enabled:

        def passthrough(fn: F) -> F:
            return fn

        return passthrough

    from langfuse import observe as _lf_observe

    return _lf_observe(*dargs, **dkwargs)
