# vlm-rag-index

A vectorless multi-document RAG system. Documents are indexed by a vision-language model that reads rendered page images and emits a structural tree; queries are answered by a LangGraph ReAct agent that navigates that tree and asks the same VLM follow-up questions on the relevant pages.

No embeddings. No chunk-and-cosine. Reasoning over structure.

## Status

Phase 0 scaffolding. The cross-cutting infrastructure (settings, logging, LLM protocol, prompt loader, PDF helpers, observability shim, CI) is in place. The VLM indexing pipeline, agent, and CLI arrive in later phases — see [docs/kickoff-project/ROADMAP.md](docs/kickoff-project/ROADMAP.md).

## Quickstart

```bash
uv sync                                     # install dependencies
cp .env.example .env                        # set LLM_API_KEY and optional Langfuse keys
uv run python -c "from vlm_rag_index.core.config import settings; print(settings.llm_model)"
```

The latter two of the original quickstart commands (`uv run index ...`, `uv run langgraph dev`) ship with Phases 1 and 2 respectively.

## Docs

Design docs live in [docs/kickoff-project/](docs/kickoff-project/). Read them in this order if starting fresh:

- [VISION.md](docs/kickoff-project/VISION.md) — why this exists.
- [ARCHITECTURE.md](docs/kickoff-project/ARCHITECTURE.md) — module layout, data flow, settings.
- [VLM_PIPELINE.md](docs/kickoff-project/VLM_PIPELINE.md) — how a PDF becomes an index.
- [AGENT.md](docs/kickoff-project/AGENT.md) — the LangGraph agent.
- [FILESYSTEM.md](docs/kickoff-project/FILESYSTEM.md) — multi-document traversal.
- [CODING_STANDARDS.md](docs/kickoff-project/CODING_STANDARDS.md) — hard rules and toolchain.
- [ROADMAP.md](docs/kickoff-project/ROADMAP.md) — phased delivery.

## License

TBD.
