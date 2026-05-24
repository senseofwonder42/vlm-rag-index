from functools import lru_cache
from pathlib import Path
from typing import Literal

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
    llm_max_output_tokens: int = 16000
    llm_max_retries: int = 10
    llm_retry_delay_s: float = 1.0
    llm_timeout: float = 120.0

    vlm_dpi: int = 144
    vlm_pages_per_batch: int = 8
    vlm_max_images_per_call: int = 8
    vlm_window_overlap: int = 0
    vlm_verify_heading_text: bool = False

    index_add_node_id: bool = True
    index_add_node_summary: bool = False
    index_add_node_text: bool = False
    index_add_doc_description: bool = False
    index_results_dir: Path = PROJECT_ROOT / "examples" / "results"

    tracing_enabled: bool = False
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: SecretStr = SecretStr("")
    langfuse_secret_key: SecretStr = SecretStr("")

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
