from vlm_rag_index.core.config import settings
from vlm_rag_index.pageindex.llm.openai_compatible_client import OpenAICompatibleClient
from vlm_rag_index.pageindex.llm.protocol import LLMClient
from vlm_rag_index.pageindex.llm.tracing_client import TracingLLMClient


def get_default_client(model: str | None = None) -> LLMClient:
    base = OpenAICompatibleClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        model=model or settings.llm_model,
        timeout=settings.llm_timeout,
        max_output_tokens=settings.llm_max_output_tokens,
        temperature=settings.llm_temperature,
        top_p=settings.llm_top_p,
        top_k=settings.llm_top_k,
        repetition_penalty=settings.llm_repetition_penalty,
        enable_thinking=settings.llm_enable_thinking,
        max_soft_tokens=settings.vlm_max_soft_tokens,
        extra_body=settings.llm_extra_body,
        max_retries=settings.llm_max_retries,
        retry_delay_s=settings.llm_retry_delay_s,
    )
    if settings.tracing_enabled:
        return TracingLLMClient(base)
    return base
