from pydantic import SecretStr

from vlm_rag_index.core.config import Settings


def test_yaml_defaults_load():
    s = Settings(llm_api_key=SecretStr("sk-test"))
    assert s.llm_model == "google/gemma-4-31b-it"
    assert s.llm_base_url == "https://openrouter.ai/api/v1"
    assert s.vlm_dpi == 144
    assert s.tracing_enabled is False


def test_init_args_win_over_yaml():
    s = Settings(llm_api_key=SecretStr("sk-test"), llm_model="explicit/model")
    assert s.llm_model == "explicit/model"


def test_env_overrides_yaml(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "from-env")
    monkeypatch.setenv("LLM_API_KEY", "sk-test-env")
    s = Settings()
    assert s.llm_model == "from-env"
    assert s.llm_api_key.get_secret_value() == "sk-test-env"
