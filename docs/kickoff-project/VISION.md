# Vision

## The vectorless thesis

Embedding-based retrieval treats relevance as cosine proximity in a learned vector space. That works for surface similarity. It breaks down when the question requires reasoning about *which part of a long structured document is actually about the thing being asked*. A 60-page technical report has dozens of sections that mention "the model"; vector RAG retrieves the wrong ones routinely.

This project bets on a different primitive: an LLM-extracted hierarchical tree of the document, navigated by a reasoning agent. The agent reads section titles, descends into the subtree that matches the question, and only at the leaf level does it look at content. The same idea humans use with a table of contents. The [PageIndex](https://github.com/VectifyAI/PageIndex) project established this approach; what follows here is its own implementation.

## VLM-first indexing

The indexer does not parse text layers or detect TOC pages. It renders every page to a PNG and sends batches of images to a vision-language model with a structured-output prompt. The model returns, for each page, the headings that start on that page, the sections that end on it, and a one-sentence content description. A small post-processing step assembles those per-page records into a tree.

Why this works:

- The VLM sees what a reader sees. Visual cues — font size, weight, spacing, indentation — disambiguate heading from body text more reliably than any text-only heuristic.
- Caption, figure, header, and author noise is filterable by prompt instruction, not by regex over fragile text extraction.
- One LLM call per batch of pages replaces the TOC-detect → TOC-parse → page-map → verify chain of a text-based pipeline. Fewer calls, fewer failure modes.
- A vision model that can index a page can also *answer questions about* the same page later in the same session. The indexer and the answerer are the same model.

## From one document to many

A single document is a tree the agent navigates. A corpus is also a tree — just one level higher. The [PageIndex Filesystem](https://pageindex.ai/blog/pageindex-filesystem) blog post lays out three mechanisms for scaling tree-search to many documents:

- **Virtual nodes**: when the on-disk file layout has no useful hierarchy, synthesize one from per-document metadata (category, entities, summary) so the agent has something to descend into. A document may live under multiple virtual ancestors.
- **Query-dependent trees**: don't fix the corpus hierarchy at ingestion. Build the view at query time, picking the metadata axes that matter for the current question.
- **Adaptive traversal**: when child labels carry signal, descend layer by layer; when they don't, flatten and defer the decision to the next level down.

The agent's reasoning policy is the same at both layers. There is no separate "corpus search" subsystem — just a bigger tree.

## Non-goals

- No embeddings. No vector database. No chunk-and-retrieve.
- No provider lock-in. Pipeline LLM calls go through a thin protocol; any OpenAI-compatible endpoint (OpenRouter, vLLM, self-hosted) works.
- No proprietary file formats. Index outputs are plain JSON; metadata sidecars are plain JSON. Anything can read them.
