from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from vlm_rag_index.core.constants import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

Environment = Literal["local", "test", "dev", "preprod", "prod"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        yaml_file=PROJECT_ROOT / "config.yaml",
        yaml_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    environment: Environment = "local"
    log_level: str = "INFO"

    llm_model: str = "google/gemma-4-31b-it"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: SecretStr = SecretStr("")
    llm_temperature: float = 0.0
    llm_top_p: float | None = None
    llm_top_k: int | None = None
    llm_repetition_penalty: float | None = None
    # Gemma emits a reasoning block by default; we don't need it for structured reads.
    llm_enable_thinking: bool = False
    # Escape hatch for vLLM sampling knobs not surfaced as typed fields (e.g. min_p,
    # presence_penalty, stop). Merged into the request body; per-call kwargs override it.
    llm_extra_body: dict[str, Any] = {}
    llm_max_output_tokens: int = 16000
    llm_max_retries: int = 10
    llm_retry_delay_s: float = 1.0
    llm_timeout: float = 120.0

    vlm_dpi: int = 144
    # Per-request vision-tower soft-token budget (mm_processor_kwargs.max_soft_tokens).
    # None leaves the server's default in place.
    vlm_max_soft_tokens: int | None = None
    vlm_pages_per_batch: int = 8
    vlm_max_images_per_call: int = 8
    vlm_window_overlap: int = 0
    vlm_verify_heading_text: bool = False

    index_add_node_id: bool = True
    index_add_node_summary: bool = False
    index_add_node_text: bool = False
    index_add_doc_description: bool = False
    index_add_metadata: bool = True
    index_results_dir: Path = PROJECT_ROOT / "examples" / "results"

    filesystem_meta_model: str | None = None
    filesystem_axes_default: list[str] = ["category", "entities"]
    filesystem_max_tree_depth: int = 3
    filesystem_flatten_threshold: float = 0.6
    filesystem_max_llm_hops_per_query: int = 5

    # Agent reasoning model (see agent/graph.py). Defaults to `llm_model`;
    # `agent_model_override` lets the orchestrator use a stronger- or faster-planning
    # model independently of the VLM/filesystem clients.
    agent_model_override: str | None = None

    # Agent context management (see agent/graph.py). Keeps the reasoning model's
    # prompt bounded across long, multi-turn conversations.
    agent_context_edit_trigger_tokens: int = 40000
    agent_context_edit_keep: int = 3
    agent_summary_trigger_tokens: int = 60000
    agent_summary_keep_messages: int = 20

    tracing_enabled: bool = False
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: SecretStr = SecretStr("")
    langfuse_secret_key: SecretStr = SecretStr("")

    @property
    def filesystem_model(self) -> str:
        """Model id for the filesystem layer's small reasoning calls.

        Defaults to `llm_model`; `filesystem_meta_model` lets a cheaper model
        handle axis selection and traversal decisions.
        """
        return self.filesystem_meta_model or self.llm_model

    @property
    def agent_model(self) -> str:
        """Model id for the LangGraph agent's reasoning loop.

        Defaults to `llm_model`; `agent_model_override` lets the orchestrator run on
        a different model than the pipeline's VLM/filesystem calls.
        """
        return self.agent_model_override or self.llm_model

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
