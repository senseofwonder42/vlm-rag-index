# Roadmap

Phased delivery. Each phase ships independently runnable software with an explicit exit criterion.

## Phase 0 — Scaffolding (1–2 days)

Make the project runnable with no business logic.

- `pyproject.toml` with the dependency set: `pydantic`, `pydantic-settings`, `pyyaml`, `loguru`, `httpx`, `pymupdf`, `pypdf2`, `jinja2`, `langgraph`, `langchain-openai`, `langfuse`; dev extras for `ruff`, `mypy`, `pytest`, `pytest-asyncio`, `pytest-env`.
- `core/{config.py,logging.py,constants.py}` — `Settings`, loguru setup, `PROJECT_ROOT`.
- `pageindex/llm/{protocol.py,openai_compatible_client.py,tracing_client.py,factory.py,retry.py}`.
- `pageindex/prompts/__init__.py` with the `render()` loader and `StrictUndefined` env.
- `pageindex/pdf/{reader.py,renderer.py}`.
- `pageindex/observability.py` — the `@observe` shim.
- `.env.example`, `config.yaml` with documented defaults.
- CI: `ruff check`, `mypy`, `pytest` jobs, all blocking.

**Touches**: [CODING_STANDARDS.md](CODING_STANDARDS.md), [ARCHITECTURE.md](ARCHITECTURE.md).

**Exit criterion**: `uv sync && uv run python -c "from <pkg>.core.config import settings; print(settings.llm_model)"` runs cleanly.

## Phase 1 — Single-doc indexing (3–5 days)

Make a PDF into a JSON.

- `pageindex/structured_responses.py` — `PageHeading`, `PageInfo`, `PageBatchResponse`, `VisualRank`.
- `pageindex/prompts/vlm_page_batch.j2`.
- `pageindex/vlm/{extract.py,reconcile.py}` — rendering, sliding window, batch call, reconciliation, tree assembly, end-index assignment.
- `pageindex/tree/{types.py,builder.py,post_process.py}` — `IndexResult`, node IDs, key reordering.
- `pageindex/pipeline.py` — `apage_index()` orchestrator.
- `cli/index.py` — `uv run index <pdf>`, plus `--input-dir`/`--output-dir` batch mode, Rich progress bar.
- Tests for `_sliding_windows`, `_reconcile_records`, `_depth_from_numbering`, end-index assignment.

**Touches**: [VLM_PIPELINE.md](VLM_PIPELINE.md), [ARCHITECTURE.md](ARCHITECTURE.md).

**Exit criterion**: indexing a 50-page PDF produces a structurally correct JSON in under 60 seconds with a vision-capable model on OpenRouter. Spot-checked against the PDF's actual table of contents.

## Phase 2 — Single-doc agent (2–3 days)

Make the JSONs queryable from a chat UI.

- `pageindex/prompts/{vlm_answer.j2,agent_system.j2}`.
- `agent/{graph.py,tools.py,vlm.py,tracing.py}` — three tools, `ChatOpenAI`, Langfuse callbacks.
- `langgraph.json` pointing at `agent/graph.py:graph`.
- Local agent-chat-ui setup verified against `uv run langgraph dev`.
- Tests for the three tools against indexed fixture JSONs.

**Touches**: [AGENT.md](AGENT.md).

**Exit criterion**: a question like "What does this paper say about ablation studies?" returns a grounded answer with page citations, via agent-chat-ui, with traces visible in Langfuse when enabled.

## Phase 3 — Per-document metadata sidecar (1–2 days)

Make the corpus filesystem-ready without exposing it yet.

- `pageindex/prompts/doc_metadata.j2`.
- `pageindex/pipeline.py` — extend `apage_index` to write `.meta.json` next to the index JSON.
- `cli/index.py` — `--no-metadata` flag to opt out for testing.
- `schema_version: 1` stamped on every sidecar from day one.
- Tests for the metadata generation step.

**Touches**: [FILESYSTEM.md](FILESYSTEM.md), [VLM_PIPELINE.md](VLM_PIPELINE.md).

**Exit criterion**: every JSON in `index_results_dir` has a corresponding `meta.json` with sensible category/entities/summary. Agent behavior unchanged from Phase 2.

## Phase 4 — Multi-doc filesystem traversal (4–6 days)

Make the corpus tree navigable.

- `pageindex/filesystem/{store.py,virtual_nodes.py,traversal.py}`.
- `agent/tools.py` — `search_filesystem`, `list_folder`, `get_folder_summary`.
- `agent_system.j2` v2 — teach search-first behavior.
- New `Settings` fields per [FILESYSTEM.md](FILESYSTEM.md).
- Tests on a fixture corpus of ≥ 20 indexed documents across ≥ 3 obvious categories.

**Touches**: [FILESYSTEM.md](FILESYSTEM.md), [AGENT.md](AGENT.md).

**Exit criterion**: a corpus-wide question ("Which document discusses Q3 revenue?") finds the right document via `search_filesystem` and answers from it via the v1 tools, in one agent run, with all hops visible in Langfuse.

## Phase 5 — Quality and ops (ongoing)

The work that doesn't have a single shipping moment.

- Heading verification (`vlm_verify_heading_text=true`) toggle wired through CLI.
- Summary toggle (`index_add_node_summary=true`) wired through CLI.
- Batch indexing with parallelism — async fan-out across multiple PDFs.
- Cost telemetry: token counts and LLM call counts per indexed doc and per agent query, surfaced in Langfuse.
- Regression test corpus: a folder of PDFs whose indexes are checked in; CI re-indexes one of them per run and diffs the tree structure.
- Deployment of the LangGraph server behind an auth proxy for non-local use.
