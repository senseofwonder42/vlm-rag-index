# Architecture

## Layered view

Three layers, with strict dependency direction:

- **`core/`** — settings, logging, project-wide constants. Knows nothing about LLMs, PDFs, or LangChain.
- **`pageindex/`** — the indexing pipeline. Reads PDFs, calls the LLM through a protocol, builds the tree, writes JSON. Depends only on `core/`.
- **`agent/`** — the LangGraph ReAct agent. Reads the JSONs produced by `pageindex/` and serves queries. Depends on `pageindex/` for renderers and types; never the reverse.

The pipeline never imports LangChain. The agent imports the pipeline's `LLMClient` only inside `answer_from_pages` (for the VLM image read), and uses `langchain_openai.ChatOpenAI` against the same endpoint for its reasoning calls. This boundary is enforced by convention; if it slips, the project is broken.

## Skeleton file tree

```
src/<pkg>/
├── core/
│   ├── __init__.py
│   ├── config.py            # Settings (pydantic-settings)
│   ├── constants.py         # PROJECT_ROOT, etc.
│   └── logging.py           # loguru setup
├── pageindex/
│   ├── __init__.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── protocol.py                  # LLMClient Protocol + Message/ContentPart
│   │   ├── openai_compatible_client.py  # httpx-only impl
│   │   ├── tracing_client.py            # Langfuse-wrapping decorator
│   │   ├── factory.py                   # get_default_client()
│   │   └── retry.py                     # exponential backoff helpers
│   ├── prompts/
│   │   ├── __init__.py                  # render(name, **ctx) loader (StrictUndefined)
│   │   ├── vlm_page_batch.j2
│   │   ├── vlm_answer.j2
│   │   ├── leaf_summary.j2
│   │   ├── parent_summary.j2
│   │   ├── doc_description.j2
│   │   ├── doc_metadata.j2              # corpus sidecar (Phase 3+)
│   │   └── agent_system.j2
│   ├── pdf/
│   │   ├── __init__.py
│   │   ├── reader.py        # PyMuPDF/PyPDF2 text + per-page tokenization
│   │   └── renderer.py      # render_page(s), image_part()
│   ├── vlm/
│   │   ├── __init__.py
│   │   ├── extract.py       # sliding-window batch + tree assembly
│   │   ├── reconcile.py     # overlap reconciliation by voting
│   │   └── summaries.py     # leaf + parent summary passes
│   ├── tree/
│   │   ├── __init__.py
│   │   ├── types.py         # TreeNode, IndexResult (pydantic)
│   │   ├── builder.py       # node_id assignment, text embedding
│   │   └── post_process.py  # strip transient fields, key reordering
│   ├── filesystem/                      # Phase 4+
│   │   ├── __init__.py
│   │   ├── store.py         # locate {pdf, json, meta.json} triples
│   │   ├── virtual_nodes.py # axis selection + tree projection
│   │   └── traversal.py     # adaptive descent / flatten policy
│   ├── pipeline.py          # apage_index() orchestrator
│   ├── structured_responses.py
│   └── observability.py     # @observe no-op shim
├── agent/
│   ├── __init__.py
│   ├── graph.py             # build_graph(); exposes `graph`
│   ├── tools.py             # v1 + v2 LangChain tools
│   ├── vlm.py               # answer_with_images()
│   └── tracing.py           # Langfuse CallbackHandler factory
└── cli/
    ├── __init__.py
    └── index.py             # `uv run index` entry point

config.yaml                  # non-secret defaults
.env.example                 # secret + tracing placeholders
langgraph.json               # points at agent/graph.py:graph
pyproject.toml
uv.lock                      # committed
tests/                       # mirrors src/ layout
docs/
```

## Data flow — indexing

```
PDF
 └─ pdf.renderer.render_pages       → {page_idx: PNG bytes}
     └─ vlm.extract._sliding_windows → list of page-index windows
         └─ for each window: vlm.extract._call_batch
             └─ llm.acomplete_structured(prompt + image_parts, PageBatchResponse)
                 → list[PageInfo]
     └─ vlm.reconcile._reconcile_records (only if window_overlap > 0)
     └─ vlm.extract._build_tree     → nested dict[str, Any]
         (depth from section numbering, falling back to visual rank;
          caption-noise filter; synthetic Front Matter if needed;
          end_index from VLM ending_sections + sibling handoff)
     └─ tree.builder.write_node_id  (if index_add_node_id)
     └─ vlm.summaries.*             (if index_add_node_summary; two passes)
     └─ tree.builder.embed_text     (if index_add_node_text)
     └─ tree.post_process.strip_internal_fields + reorder_keys
     └─ doc_description LLM call    (if index_add_doc_description)
 → IndexResult JSON, written beside the source PDF
 → doc_metadata LLM call → .meta.json beside the PDF (Phase 3+)
```

## Data flow — querying

```
agent-chat-ui ──HTTP──▶ langgraph dev server ──▶ agent.graph (create_react_agent)
                                                  │
                                                  ├─ ChatOpenAI (reasoning)
                                                  ├─ tools.list_documents
                                                  ├─ tools.get_document_structure
                                                  ├─ tools.answer_from_pages
                                                  │   ├─ pdf.renderer.render_pages
                                                  │   └─ agent.vlm.answer_with_images
                                                  │       └─ llm.acomplete(text + image_parts)
                                                  └─ tools.search_filesystem (v2)
                                                      └─ pageindex.filesystem.traversal
```

## Settings layering

`core/config.py` defines a `Settings` class on `pydantic-settings`. Loading priority, highest first:

1. Init arguments (passed to `Settings(...)` directly — used in tests)
2. Environment variables
3. `.env` file
4. `config.yaml` in the project root
5. Field defaults

Every env var the project reads is a field on `Settings`. Never call `os.environ` or `os.getenv` anywhere else.

The `Settings` singleton is exposed as `from <pkg>.core.config import settings`. Tests construct `Settings(...)` directly when they need overrides.

## Settings summary

| Setting | Default | Effect |
|---|---|---|
| `environment` | `local` | One of `local`, `test`, `dev`, `preprod`, `prod`. |
| `log_level` | `INFO` | Loguru level. |
| `llm_model` | (required) | OpenAI-compatible model id. Must be vision-capable. |
| `llm_base_url` | `https://openrouter.ai/api/v1` | OpenAI-compatible endpoint. |
| `llm_api_key` | (secret, from `.env`) | Bearer credential. |
| `llm_temperature` | `0.0` | Sampling temperature for all pipeline calls. |
| `llm_max_output_tokens` | `16000` | Default max output tokens per call. |
| `llm_max_retries` | `10` | httpx-level retry budget. |
| `llm_retry_delay_s` | `1.0` | Base delay for exponential backoff. |
| `llm_timeout` | `120.0` | Per-call timeout (seconds). |
| `vlm_dpi` | `144` | Render DPI; long edge ~1568px on A4. |
| `vlm_pages_per_batch` | `8` | Images per VLM call. |
| `vlm_max_images_per_call` | `8` | Provider-side cap; effective batch is `min(per_batch, max_per_call)`. |
| `vlm_window_overlap` | `0` | Sliding-window overlap (pages). |
| `vlm_verify_heading_text` | `false` | Snap VLM headings to closest text-layer line. |
| `index_add_node_id` | `true` | Depth-first zero-padded node IDs. |
| `index_add_node_summary` | `false` | Leaf + parent summaries (extra LLM calls). |
| `index_add_node_text` | `false` | Embed raw page text in leaf nodes. |
| `index_add_doc_description` | `false` | One-sentence root summary. Requires `index_add_node_summary`. |
| `index_results_dir` | `./examples/results` | Where `{pdf, json, meta.json}` triples live. |
| `tracing_enabled` | `false` | Route LLM + agent calls through Langfuse when true. |
| `langfuse_host` | `http://localhost:3000` | Self-hosted or cloud Langfuse. |
| `langfuse_public_key` | (secret) | Langfuse public key. |
| `langfuse_secret_key` | (secret) | Langfuse secret key. |

Multi-document settings (`filesystem_*`) appear in [FILESYSTEM.md](FILESYSTEM.md) and ship in Phase 4.

## Entry points

- **`uv run index <pdf>`** — indexes one PDF. `--input-dir <dir>` for batch mode; `--output-dir <dir>` to write JSONs elsewhere than next to the PDFs. Entry point is `cli/index.py:main`, declared in `pyproject.toml`'s `[project.scripts]`.
- **`uv run langgraph dev`** — LangGraph dev server reading `langgraph.json`. Serves the compiled graph on `http://localhost:2024` by default.
- **`agent-chat-ui`** — external React app, pointed at the LangGraph dev server URL. Not bundled in this repo.
