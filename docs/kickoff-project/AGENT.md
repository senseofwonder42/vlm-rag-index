# Agent layer

A LangGraph ReAct agent that exposes indexed documents as tools and serves a chat UI via `agent-chat-ui`.

## Stack

- `langgraph` for the graph runtime and `langgraph dev` server.
- `langchain-openai` (`ChatOpenAI`) for the agent's reasoning calls, pointed at the same OpenAI-compatible endpoint the pipeline uses.
- `langfuse` for tracing, via LangChain `CallbackHandler`.
- [agent-chat-ui](https://github.com/langchain-ai/agent-chat-ui) as the frontend — external project, not bundled here, configured against the dev server URL.

The agent never imports the pipeline's `LLMClient` for reasoning. The pipeline never imports `langchain*`. LangChain is contained to `agent/`; the only place a `ChatOpenAI` exists is `agent/graph.py`. The agent does call into `pageindex/llm` once — inside `answer_from_pages` — to issue a VLM read against rendered page images, because the same image-call infrastructure used at indexing time is exactly what we want there. That is the one allowed cross-layer call.

## Graph

`agent/graph.py:build_graph()` returns a compiled `StateGraph` via `langgraph.prebuilt.create_react_agent`:

```python
graph = create_react_agent(
    model=ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        temperature=settings.llm_temperature,
    ),
    tools=TOOLS,
    prompt=render("agent_system.j2"),
)
```

`langgraph.json` exposes it:

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/<pkg>/agent/graph.py:graph"
  },
  "env": ".env"
}
```

`uv run langgraph dev` serves the graph on `http://localhost:2024`.

## System prompt

Lives as `prompts/agent_system.j2` and is rendered once at graph build, not per request. It tells the model:

- It is answering questions about PDF documents that have been pre-indexed.
- It must always pick a document first (`list_documents` if it doesn't know what's available) before asking content questions.
- It should call `get_document_structure` to find which pages are relevant, then `answer_from_pages` with a tight page range — never more than ~20 pages.
- It should cite page numbers in its answer.

Prompts are Jinja2 templates under `pageindex/prompts/` and rendered via `prompts.render("agent_system.j2", **ctx)` — same loader as the pipeline.

## v1 tools (single-document)

All three are `async` functions decorated with `langchain.tools.tool` and live in `agent/tools.py`.

### `list_documents() -> list[str]`

Returns the `doc_name` field of every `*.json` in `settings.index_results_dir` that has a matching `*.pdf` next to it. Documents without a matching PDF are skipped — `answer_from_pages` would fail anyway.

### `get_document_structure(doc_name: str) -> dict`

Reads the `IndexResult` JSON for `doc_name`, **strips the `text` field from every node** (it is huge and the agent doesn't need it for navigation), and returns the trimmed tree. Includes `node_id`, `title`, `start_index`, `end_index`, `summary` (if present), and `nodes`. Returns an error message string when `doc_name` is not found.

### `answer_from_pages(doc_name: str, page_indices: list[int], question: str) -> str`

The leaf retrieval call. Steps:

1. Resolve `<doc_name>.pdf` inside `settings.index_results_dir`.
2. `render_pages(pdf, page_indices, dpi=settings.vlm_dpi)` → `{page_idx: PNG bytes}`.
3. Build a single user message: rendered `prompts/vlm_answer.j2` text + one `image_part` per requested page.
4. `llm.acomplete(messages)` — *not* `acomplete_structured`. We want free-text.
5. Return the stripped response text.

The VLM call uses the same `LLMClient` factory the pipeline uses (`llm.factory.get_default_client()`), so it inherits tracing through `TracingLLMClient` when enabled. This is the one place agent code reaches into the pipeline's LLM stack.

`vlm_answer.j2` instructs the model to answer the question grounded in the provided page images, cite page numbers, and say "I don't know from these pages" when the images don't contain the answer. No retrieval-augmentation gymnastics; the images *are* the context.

## v2 tools (multi-document)

These ship with the filesystem layer. Sketched here; specced in [FILESYSTEM.md](FILESYSTEM.md).

- `search_filesystem(query: str) -> list[str]` — adaptive traversal of the corpus tree; returns candidate `doc_name`s ranked by relevance.
- `list_folder(path: str = "/") -> list[dict]` — children of a (possibly virtual) folder in the current query-dependent corpus tree.
- `get_folder_summary(path: str) -> str` — synthesized description of a folder's contents.

The v2 system prompt teaches the agent to start with `search_filesystem` when the question is corpus-wide. `list_documents` stays available as the dumb fallback.

## Image content parts

`pdf/renderer.image_part(png: bytes) -> ImageUrlPart`:

```python
def image_part(png: bytes) -> ImageUrlPart:
    b64 = base64.b64encode(png).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
```

OpenAI-style. Supported by OpenAI, Anthropic-via-compat, and most OpenRouter providers. One helper, two callers — the pipeline at indexing time and `answer_from_pages` at query time.

## Tracing

`agent/tracing.py:langchain_callbacks() -> list[BaseCallbackHandler]` returns `[CallbackHandler(public_key=..., secret_key=..., host=...)]` when `settings.tracing_enabled` is true, else `[]`. The graph is configured per request with `RunnableConfig(callbacks=langchain_callbacks())`.

Pipeline calls — indexing and the `answer_from_pages` VLM read — trace through `TracingLLMClient`, which logs Langfuse generations directly. Both surfaces land in the same Langfuse project. One trace per agent request, with the VLM reads as child spans.

When tracing is off, `langchain_callbacks()` returns an empty list and the `@observe` shim in `observability.py` short-circuits — zero overhead, no missing-credentials warnings.

## agent-chat-ui integration

`agent-chat-ui` is a separate Next.js project. Point it at the LangGraph dev server:

```
NEXT_PUBLIC_API_URL=http://localhost:2024
NEXT_PUBLIC_ASSISTANT_ID=agent
```

No auth in local dev. Production deployment of the API server is out of scope for v1; if needed, front it with an auth proxy and pass a header through the UI's `apiKey` setting.
