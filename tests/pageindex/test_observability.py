from vlm_rag_index.pageindex import observability


def test_observe_is_passthrough_when_tracing_disabled(monkeypatch):
    # Default Settings has tracing_enabled=False; the module-level `settings` reflects that.
    assert observability.settings.tracing_enabled is False

    @observability.observe()
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5
    # Passthrough must return the original function object, not a wrapper.
    assert add.__name__ == "add"
