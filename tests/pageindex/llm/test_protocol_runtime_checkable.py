from typing import Any

from vlm_rag_index.pageindex.llm.protocol import LLMClient


class _StubClient:
    def complete(self, messages, **kwargs):  # noqa: ARG002
        return ""

    async def acomplete(self, messages, **kwargs):  # noqa: ARG002
        return ""

    def complete_structured(self, messages, schema, **kwargs):  # noqa: ARG002
        return schema()

    async def acomplete_structured(self, messages, schema, **kwargs):  # noqa: ARG002
        return schema()

    def count_tokens(self, text):  # noqa: ARG002
        return 0


def test_stub_satisfies_protocol():
    client: Any = _StubClient()
    assert isinstance(client, LLMClient)


def test_empty_object_does_not_satisfy_protocol():
    class _Empty:
        pass

    obj: Any = _Empty()
    assert not isinstance(obj, LLMClient)
