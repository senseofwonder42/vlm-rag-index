from vlm_rag_index.pageindex import observability
from vlm_rag_index.pageindex.observability import _trace_safe_messages


def test_trace_safe_keeps_text_and_offloads_images():
    messages = [
        {"role": "system", "content": "you are helpful"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "read this page"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        },
    ]

    safe = _trace_safe_messages(messages, lambda url: {"media": url})

    # String content is passed through unchanged.
    assert safe[0] == {"role": "system", "content": "you are helpful"}
    # Text part kept verbatim; image data-URI routed through the media factory.
    assert safe[1]["content"][0] == {"type": "text", "text": "read this page"}
    assert safe[1]["content"][1] == {
        "type": "image",
        "data": {"media": "data:image/png;base64,AAAA"},
    }


def test_observe_is_passthrough_when_tracing_disabled(monkeypatch):
    # Default Settings has tracing_enabled=False; the module-level `settings` reflects that.
    assert observability.settings.tracing_enabled is False

    @observability.observe()
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5
    # Passthrough must return the original function object, not a wrapper.
    assert add.__name__ == "add"
