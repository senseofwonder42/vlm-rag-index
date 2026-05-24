# &lt;project-name&gt;

A vectorless multi-document RAG system. Documents are indexed by a vision-language model that reads rendered page images and emits a structural tree; queries are answered by a LangGraph ReAct agent that navigates that tree and asks the same VLM follow-up questions on the relevant pages.

No embeddings. No chunk-and-cosine. Reasoning over structure.

## Status

v0 scaffolding. Single-document indexing and agent are the v1 milestone; multi-document filesystem traversal is designed in from day one and ships in v2.

## Quickstart

```bash
uv sync                                     # install dependencies
cp .env.example .env                        # set LLM_API_KEY and optional Langfuse keys
uv run index path/to/doc.pdf                # produces doc.json next to the PDF
uv run langgraph dev                        # start the agent API server (port 2024)
```

Then point [agent-chat-ui](https://github.com/langchain-ai/agent-chat-ui) at `http://localhost:2024` with assistant id `agent`.

## Docs

- [VISION.md](VISION.md) — why this exists and what we are betting on.
- [ARCHITECTURE.md](ARCHITECTURE.md) — module layout, data flow, skeleton file tree, settings table.
- [VLM_PIPELINE.md](VLM_PIPELINE.md) — how a PDF becomes an index, stage by stage.
- [AGENT.md](AGENT.md) — LangGraph ReAct agent, tool surface, agent-chat-ui integration.
- [FILESYSTEM.md](FILESYSTEM.md) — multi-document tree navigation (virtual nodes, query-dependent trees, adaptive traversal).
- [CODING_STANDARDS.md](CODING_STANDARDS.md) — hard rules and toolchain.
- [ROADMAP.md](ROADMAP.md) — phased delivery from scaffolding to multi-doc.

Read them in that order if starting fresh.

## License

TBD.
