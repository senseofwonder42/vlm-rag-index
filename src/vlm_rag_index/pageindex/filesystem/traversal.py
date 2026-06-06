"""Adaptive descent over the query-dependent corpus tree.

The tree is never stored. For each query we pick a metadata axis, project the
candidates into folders, and ask the LLM which folders to keep. If the folder
labels don't discriminate (low confidence), we *flatten* — keep every folder and
defer the decision to the next axis. The recursion is bounded by
`filesystem_max_tree_depth` and a hard per-query LLM-call budget.
"""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel, Field

from vlm_rag_index.core.config import Settings
from vlm_rag_index.pageindex.filesystem import virtual_nodes
from vlm_rag_index.pageindex.filesystem.store import CorpusDoc
from vlm_rag_index.pageindex.llm.protocol import LLMClient, Message
from vlm_rag_index.pageindex.prompts import render


class RelevanceDecision(BaseModel):
    """Which folders the LLM judged relevant, and how sure it is."""

    relevant_labels: list[str] = Field(
        description="Folder names worth keeping, copied verbatim.",
    )
    confidence: float = Field(
        description="0..1 — how clearly the folder names discriminate relevance.",
    )


async def _decide(
    client: LLMClient, query: str, axis: str, tree: dict[str, list[str]]
) -> RelevanceDecision:
    folders = [(label, len(docs)) for label, docs in tree.items()]
    prompt = render("fs_relevance.j2", query=query, axis=axis, folders=folders)
    messages: list[Message] = [{"role": "user", "content": prompt}]
    return await client.acomplete_structured(messages, RelevanceDecision)


async def search_filesystem(
    query: str,
    corpus: list[CorpusDoc],
    *,
    client: LLMClient,
    settings: Settings,
) -> list[str]:
    """Navigate the corpus toward documents relevant to `query`.

    Returns candidate `doc_name`s. Descends one metadata axis per level, keeping
    the folders the LLM deems relevant and flattening (keeping all) when its
    confidence falls below `filesystem_flatten_threshold`. Stops at the depth or
    hop-budget cap and returns whatever candidates remain.
    """
    candidates = list(corpus)
    axes = virtual_nodes.available_axes(corpus, settings.filesystem_axes_default)
    hops = settings.filesystem_max_llm_hops_per_query
    depth = 0

    while depth < settings.filesystem_max_tree_depth and len(candidates) > 1 and axes:
        if hops <= 0:
            logger.debug("filesystem hop budget exhausted at depth {}", depth)
            break
        axis = await virtual_nodes.select_axis(client, query, axes, candidates)
        hops -= 1
        axes = [a for a in axes if a != axis]
        tree = virtual_nodes.project(candidates, axis)
        if len(tree) < 2:
            continue  # axis does not split this candidate set; try another
        if hops <= 0:
            break
        decision = await _decide(client, query, axis, tree)
        hops -= 1

        flatten = decision.confidence < settings.filesystem_flatten_threshold
        labels = (
            list(tree)
            if flatten
            else [lbl for lbl in decision.relevant_labels if lbl in tree]
        )
        if not labels:
            break  # nothing relevant on this axis; keep current candidates

        keep = {name for label in labels for name in tree[label]}
        narrowed = [d for d in candidates if d.doc_name in keep]
        if not narrowed:
            break
        candidates = narrowed
        depth += 1

    return [d.doc_name for d in candidates]
