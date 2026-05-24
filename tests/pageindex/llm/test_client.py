import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from vlm_rag_index.pageindex.llm.openai_compatible_client import OpenAICompatibleClient


class _Toy(BaseModel):
    name: str
    score: int


def _canned_response(content: str, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        json={"choices": [{"message": {"content": content}}]} if status == 200 else {"error": "x"},
        request=httpx.Request("POST", "https://example.test/chat/completions"),
    )


def _client(transport: httpx.MockTransport) -> OpenAICompatibleClient:
    c = OpenAICompatibleClient(
        base_url="https://example.test",
        api_key="sk-test",
        model="test/model",
        timeout=5.0,
        max_output_tokens=128,
        temperature=0.0,
        max_retries=3,
        retry_delay_s=0.0,
    )
    # Swap the underlying httpx clients for ones that route through the mock transport.
    c._client = httpx.Client(transport=transport, base_url="https://example.test", headers=c._headers)
    c._aclient = httpx.AsyncClient(
        transport=transport, base_url="https://example.test", headers=c._headers
    )
    return c


def test_acomplete_returns_content():
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode()))
        return _canned_response("hello world")

    import asyncio

    c = _client(httpx.MockTransport(handler))
    out = asyncio.run(c.acomplete([{"role": "user", "content": "hi"}]))
    assert out == "hello world"
    assert captured[0]["model"] == "test/model"
    assert captured[0]["messages"][0]["role"] == "user"
    assert captured[0]["max_tokens"] == 128


def test_acomplete_structured_parses_pydantic():
    payload = json.dumps({"name": "alice", "score": 7})

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["name"] == "_Toy"
        return _canned_response(payload)

    import asyncio

    c = _client(httpx.MockTransport(handler))
    out = asyncio.run(c.acomplete_structured([{"role": "user", "content": "x"}], _Toy))
    assert out == _Toy(name="alice", score=7)


def test_acomplete_retries_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(
                status_code=503,
                json={"error": "busy"},
                request=request,
            )
        return _canned_response("ok")

    import asyncio

    c = _client(httpx.MockTransport(handler))
    out = asyncio.run(c.acomplete([{"role": "user", "content": "hi"}]))
    assert out == "ok"
    assert calls["n"] == 3


def test_complete_sync_path():
    def handler(request: httpx.Request) -> httpx.Response:
        return _canned_response("sync-ok")

    c = _client(httpx.MockTransport(handler))
    assert c.complete([{"role": "user", "content": "x"}]) == "sync-ok"


def test_non_retryable_4xx_raises_immediately():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            status_code=400,
            json={"error": "bad"},
            request=request,
        )

    import asyncio

    c = _client(httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(c.acomplete([{"role": "user", "content": "x"}]))
    assert calls["n"] == 1


def test_count_tokens_heuristic():
    c = OpenAICompatibleClient(
        base_url="https://example.test", api_key="x", model="m"
    )
    assert c.count_tokens("") == 1
    assert c.count_tokens("a" * 8) == 2
