# Coding standards

Conventions the project lives by, enforced by tooling where possible.

## Toolchain

- **Python ≥ 3.12**.
- **`uv`** is the only sanctioned package manager. Use `uv sync` after checkout, `uv add <pkg>` (or `uv add --dev <pkg>`) to add a dependency, `uv remove <pkg>` to drop one, `uv run <cmd>` to run anything in the venv. Never `uv pip install`. Never manage the venv by hand.
- **`ruff`** for lint and format. Line length **105**, indent 4. Active rule sets: `E, W, F, I, B, PTH, ANN, ARG`.
  - `ANN`: every function takes type annotations on its arguments and return type. Tests are exempt.
  - `PTH`: prefer `pathlib.Path` over `os.path`. No exceptions in new code.
  - `ARG`: unused function arguments must be `_`-prefixed; tests are exempt.
- **`mypy`** with `pydantic.mypy`, `check_untyped_defs = true`, `no_implicit_optional = true`. Scans `src/` only. Tests are not type-checked.
- **`pytest`** with `pytest-asyncio` and `pytest-env` (injects `ENVIRONMENT=test` and `LOG_LEVEL=INFO`).

Single source of truth: `pyproject.toml`. Reproducible builds: `uv.lock` is committed.

## Hard rules

Non-negotiable. Breaking one is a bug, not a style nit.

### 1. Pipeline LLM calls go through the `LLMClient` Protocol

`pageindex/llm/protocol.py` defines a `runtime_checkable` `LLMClient` Protocol with `complete`, `acomplete`, `complete_structured`, `acomplete_structured`, `count_tokens`. The default implementation, `OpenAICompatibleClient`, uses raw `httpx` — not the `openai` SDK. To add a new provider, implement the protocol; do not extend the OpenAI-compatible client with provider-specific branches.

The agent layer is the one place that uses `langchain-openai`'s `ChatOpenAI`. Pipeline code never imports `langchain*`. Agent code reaches into the pipeline's `LLMClient` only inside `answer_from_pages` — for the VLM image read, not for reasoning.

### 2. Prompts live in `.j2` files

Every pipeline prompt (and the agent system prompt) is a Jinja2 template under `pageindex/prompts/`, rendered via `prompts.render("name.j2", **ctx)`. The Jinja environment uses `StrictUndefined` — referencing an unprovided variable raises rather than silently producing the empty string. No multiline prompts as Python f-strings. No f-string prompt assembly. No inline `"...".format(...)`.

### 3. Loguru only

`from loguru import logger`. No `print()` for diagnostics. No `import logging`. Loguru is configured once in `core/logging.py:setup_logging(level)`, called at process start.

### 4. In-flight pipeline state uses TypedDicts; outputs use Pydantic

Tree assembly during indexing operates on `dict[str, Any]` (or named TypedDicts when worth it). The pipeline's *output* — `IndexResult`, `TreeNode`, the structured-response models — uses Pydantic. Don't convert internal dicts to Pydantic mid-pipeline; the validation cost compounds and adds no safety.

### 5. `@observe` comes from `observability.py`

`pageindex/observability.py` provides an `@observe` decorator that is a no-op when `settings.tracing_enabled` is false and forwards to Langfuse's `@observe` when true. Never `from langfuse import observe` directly — the shim suppresses Langfuse's noisy "missing credentials" warnings when tracing is off, which is the default in development.

### 6. No `os.environ` reads outside `Settings`

Every environment variable the project reads is a field on the `Settings` class in `core/config.py`. Code reads `settings.foo`, never `os.environ["FOO"]` or `os.getenv("FOO")`.

### 7. `pathlib` over `os.path`

Enforced by ruff `PTH`. Functions that take file paths accept `Path | str` and immediately convert with `Path(...)`. Functions that return paths return `Path`.

### 8. Type annotations everywhere

Enforced by ruff `ANN`. Every function — top-level, method, nested — has annotated parameters and a return type. Tests are exempt. `Any` is allowed but a last resort.

## Conventions (expected, not enforced)

- **Module size**: prefer many small modules over a few large ones. The `pageindex/` package has ~10 subdirectories; that is intentional.
- **No code in `__init__.py`** except `from .module import name` re-exports for the public API.
- **Tests live next to behavior**. `tests/pageindex/vlm/test_extract.py` mirrors `src/<pkg>/pageindex/vlm/extract.py`.
- **Async by default in the pipeline**. The orchestrator is `apage_index`; a synchronous `page_index` wrapper exists for CLI ergonomics.
- **Settings are immutable in a process**. `core/config.py` exposes a `@lru_cache`d `get_settings()` and a module-level `settings = get_settings()`. Tests that need different values use `Settings(...)` direct construction in fixtures.
- **No print debugging in committed code**. If you needed a `print`, you needed a `logger.debug`. Add it, gate it on `LOG_LEVEL=DEBUG`, and leave it.

## CI

Three blocking jobs from day one: `uv run ruff check`, `uv run mypy`, `uv run pytest`. No optional checks. A red CI run blocks merge.

## Commits

- [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, `perf:`, `test:`.
- Imperative mood, sentence case, no trailing period in the subject line.
- One logical change per commit; the diff should be reviewable in under five minutes.
