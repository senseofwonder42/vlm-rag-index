# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Vectorless multi-document RAG. A vision-language model reads rendered PDF page images and emits a structural tree (`IndexResult` JSON); a LangGraph ReAct agent navigates that tree and asks the same VLM follow-up questions on the relevant pages. No embeddings, no chunking — reasoning over document structure.

> The `docs/kickoff-project/` design docs (ARCHITECTURE, AGENT, FILESYSTEM, VLM_PIPELINE, CODING_STANDARDS, ROADMAP) are the authoritative spec and are detailed. They describe the intended design; some have drifted from the code (e.g. `agent/graph.py` now uses `langchain.agents.create_agent` with context-editing/summarization middleware, not `create_react_agent`; `config.yaml` ships `vlm_pages_per_batch: 1`). When docs and code disagree, the code wins — but read the docs first for the "why."

## Commands

All commands run through `uv` (the only sanctioned package manager — never `pip`/`uv pip`, never hand-edit `pyproject.toml` deps; use `uv add` / `uv add --dev` / `uv remove`).

```bash
uv sync                              # install deps after checkout
uv run ruff check                    # lint (blocking in CI)
uv run ruff format                   # format (line length 105)
uv run mypy                          # type-check src/ only (blocking in CI)
uv run pytest                        # full test suite (blocking in CI)
uv run pytest tests/pageindex/vlm/test_extract.py            # one file
uv run pytest tests/eval/test_metrics.py::test_no_calls_hits_neither   # one test
```

The three CI jobs are `ruff check`, `mypy`, `pytest` — all blocking, no optional checks.

### Application entry points (declared in `pyproject.toml` `[project.scripts]`)

```bash
uv run index <pdf>                   # index one PDF → <stem>.json (+ .meta.json) beside it
uv run index --input-dir <dir>      # batch mode; --output-dir to write elsewhere
uv run langgraph dev                 # serve the agent graph (langgraph.json) on :2024
```

### Eval workflow (operates on `dataset/`, in this order)

```bash
uv run subsample                     # build dataset/sample/ — a few PDFs per domain for quick runs
uv run inject                        # symlink PDFs into dataset/index/ + write .json/.meta.json there
uv run ask "<question>"              # ask the agent once; prints answer + which doc/pages it read
uv run eval-pages                    # score page-retrieval accuracy vs dataset/manifest.jsonl
```

`inject` keeps `dataset/pdfs/` pristine by symlinking each PDF into a sibling `dataset/index/` and writing the index triple there — the store resolves `<stem>.pdf`/`.json`/`.meta.json` by shared stem within one directory, so all three must live together.

## Architecture — three layers, strict dependency direction

- **`core/`** — `config.py` (pydantic-settings), `logging.py` (loguru), `constants.py`. Knows nothing about LLMs/PDFs/LangChain.
- **`pageindex/`** — the indexing pipeline. Reads PDFs, calls the LLM through a protocol, builds/summarizes the tree, writes JSON. Depends only on `core/`. **Never imports `langchain*`.**
- **`agent/`** — the LangGraph ReAct agent. Reads the JSONs `pageindex/` produced and serves queries. Depends on `pageindex/` for renderers/types — never the reverse.

The boundary that defines the project: the **pipeline never imports LangChain**; the **agent uses `langchain_openai.ChatOpenAI` only for reasoning** (the sole `ChatOpenAI` lives in `agent/graph.py`). The one allowed cross-layer call is `agent/tools.py:answer_from_pages` reaching into `pageindex/llm` to issue a VLM image read — reusing the indexing-time image infrastructure. If this slips, the project is broken.

### Indexing data flow (`pageindex/pipeline.py:apage_index`)

PDF → `pdf/renderer.render_pages` (PNG per page) → `vlm/extract` sliding-window batches → `llm.acomplete_structured` → `list[PageInfo]` → (`vlm/reconcile` if overlap) → `_build_tree` → optional `tree/builder` node-ids/text + `vlm/summaries` (two passes) → `tree/post_process` strip/reorder → `IndexResult` JSON. Optional `agenerate_metadata` writes the `.meta.json` sidecar (the multi-doc corpus index — see `docs/kickoff-project/FILESYSTEM.md`).

### Query data flow

agent-chat-ui → `langgraph dev` → `agent/graph.py:graph` (`create_agent` + ChatOpenAI + middleware) → tools: `list_documents`, `get_document_structure` (strips bulky `text` fields), `answer_from_pages` (renders pages → VLM read), plus v2 filesystem-traversal tools.

## Hard rules (from `docs/kickoff-project/CODING_STANDARDS.md`)

These are bugs when broken, not style nits:

1. **Pipeline LLM calls go through the `LLMClient` Protocol** (`pageindex/llm/protocol.py`). Default `OpenAICompatibleClient` uses raw `httpx`, not the `openai` SDK. New providers implement the protocol — don't branch the OpenAI client.
2. **Prompts are `.j2` files** under `pageindex/prompts/`, rendered via `prompts.render("name.j2", **ctx)`. The Jinja env uses `StrictUndefined`. No prompt f-strings or `.format()`.
3. **Loguru only** — `from loguru import logger`. No `print()`, no `import logging`.
4. **In-flight pipeline state is `dict`/TypedDict; outputs are Pydantic.** Don't convert internal dicts to Pydantic mid-pipeline.
5. **`@observe` comes from `pageindex/observability.py`**, never `from langfuse import observe` — the shim is a no-op when tracing is off and suppresses Langfuse credential warnings.
6. **No `os.environ`/`os.getenv` outside `Settings`.** Every env var is a field on `Settings` in `core/config.py`; read `settings.foo`.
7. **`pathlib` over `os.path`** (ruff `PTH`). Path params take `Path | str` and convert immediately; return `Path`.
8. **Type annotations everywhere** (ruff `ANN`) — every function's params and return. Tests exempt. `Any` is a last resort (`ANN401` is the one ignored rule).

### Conventions

- Settings are immutable per-process: `core/config.py` exposes `@lru_cache get_settings()` and a module-level `settings = get_settings()`. Tests override by constructing `Settings(...)` directly in fixtures. Settings layering, highest priority first: init args → env vars → `.env` → `config.yaml` → field defaults.
- Async by default in the pipeline (`apage_index`); sync wrappers exist for CLI ergonomics.
- Tests mirror `src/` layout (`tests/pageindex/vlm/test_extract.py` ↔ `src/.../vlm/extract.py`). `unittest` is banned — `pytest` only. `asyncio_mode = auto`.
- No code in `__init__.py` beyond re-export imports.
- Conventional Commits, imperative mood, sentence case, no trailing period.

## Local environment note

The local `.env` sets `TRACING_ENABLED=true`, which makes `tests/core/test_config.py::test_yaml_defaults_load` and `tests/pageindex/test_observability.py` fail (they assert the `false` default). This is an environment artifact, not a code regression — verify against a clean settings construction, not the ambient `.env`.
