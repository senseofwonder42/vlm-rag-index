from typing import Any, Callable, TypeVar

from vlm_rag_index.core.config import settings

F = TypeVar("F", bound=Callable[..., Any])


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
