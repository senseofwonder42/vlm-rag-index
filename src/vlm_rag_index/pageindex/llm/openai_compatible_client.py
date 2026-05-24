import asyncio
import time
from typing import Any, TypeVar

import httpx
from loguru import logger
from pydantic import BaseModel

from vlm_rag_index.pageindex.llm.protocol import Message
from vlm_rag_index.pageindex.llm.retry import exponential_backoff

T = TypeVar("T", bound=BaseModel)

_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class OpenAICompatibleClient:
    """Raw httpx-based client targeting any OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        max_output_tokens: int = 16000,
        temperature: float = 0.0,
        max_retries: int = 10,
        retry_delay_s: float = 1.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._max_retries = max_retries
        self._retry_delay_s = retry_delay_s
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(base_url=self._base_url, headers=self._headers, timeout=timeout)
        self._aclient = httpx.AsyncClient(
            base_url=self._base_url, headers=self._headers, timeout=timeout
        )

    def _build_payload(
        self,
        messages: list[Message],
        schema: type[BaseModel] | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_output_tokens,
        }
        payload.update(kwargs)
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": True,
                },
            }
        return payload

    def _extract_content(self, data: dict[str, Any]) -> str:
        return data["choices"][0]["message"]["content"]

    def complete(self, messages: list[Message], **kwargs: Any) -> str:
        payload = self._build_payload(messages, schema=None, **kwargs)
        data = self._request_sync(payload)
        return self._extract_content(data)

    async def acomplete(self, messages: list[Message], **kwargs: Any) -> str:
        payload = self._build_payload(messages, schema=None, **kwargs)
        data = await self._request_async(payload)
        return self._extract_content(data)

    def complete_structured(
        self,
        messages: list[Message],
        schema: type[T],
        **kwargs: Any,
    ) -> T:
        payload = self._build_payload(messages, schema=schema, **kwargs)
        data = self._request_sync(payload)
        content = self._extract_content(data)
        return schema.model_validate_json(content)

    async def acomplete_structured(
        self,
        messages: list[Message],
        schema: type[T],
        **kwargs: Any,
    ) -> T:
        payload = self._build_payload(messages, schema=schema, **kwargs)
        data = await self._request_async(payload)
        content = self._extract_content(data)
        return schema.model_validate_json(content)

    def count_tokens(self, text: str) -> int:
        # Char-based heuristic. Used only for diagnostic logging, never for hard budgeting.
        return max(1, len(text) // 4)

    def _handle_response(self, response: httpx.Response) -> dict[str, Any] | None:
        """Return parsed JSON when terminal-success; None when caller should retry.

        Raises on non-retryable failure (4xx other than the listed transient codes).
        """
        if response.status_code in _RETRYABLE_STATUS:
            return None
        response.raise_for_status()
        return response.json()

    def _request_sync(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_response: httpx.Response | None = None
        last_transport: httpx.TransportError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post("/chat/completions", json=payload)
            except httpx.TransportError as exc:
                last_transport = exc
                if attempt >= self._max_retries:
                    raise
                delay = exponential_backoff(attempt, base=self._retry_delay_s)
                logger.warning(
                    "LLM transport error model={} attempt={} delay={:.2f}s err={}",
                    self._model,
                    attempt,
                    delay,
                    exc,
                )
                time.sleep(delay)
                continue
            result = self._handle_response(response)
            if result is not None:
                return result
            last_response = response
            if attempt >= self._max_retries:
                break
            delay = exponential_backoff(attempt, base=self._retry_delay_s)
            logger.warning(
                "LLM retryable status model={} status={} attempt={} delay={:.2f}s",
                self._model,
                response.status_code,
                attempt,
                delay,
            )
            time.sleep(delay)
        if last_response is not None:
            last_response.raise_for_status()
        assert last_transport is not None
        raise last_transport

    async def _request_async(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_response: httpx.Response | None = None
        last_transport: httpx.TransportError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._aclient.post("/chat/completions", json=payload)
            except httpx.TransportError as exc:
                last_transport = exc
                if attempt >= self._max_retries:
                    raise
                delay = exponential_backoff(attempt, base=self._retry_delay_s)
                logger.warning(
                    "LLM transport error model={} attempt={} delay={:.2f}s err={}",
                    self._model,
                    attempt,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
                continue
            result = self._handle_response(response)
            if result is not None:
                return result
            last_response = response
            if attempt >= self._max_retries:
                break
            delay = exponential_backoff(attempt, base=self._retry_delay_s)
            logger.warning(
                "LLM retryable status model={} status={} attempt={} delay={:.2f}s",
                self._model,
                response.status_code,
                attempt,
                delay,
            )
            await asyncio.sleep(delay)
        if last_response is not None:
            last_response.raise_for_status()
        assert last_transport is not None
        raise last_transport
