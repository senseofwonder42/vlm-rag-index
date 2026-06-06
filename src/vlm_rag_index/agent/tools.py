"""LangChain tools the ReAct agent calls.

Two families:

- v1 (single-document): `list_documents`, `get_document_structure`,
  `answer_from_pages` — operate on a document the agent already chose.
- v2 (multi-document): `search_filesystem`, `list_folder`, `get_folder_summary`
  — navigate the corpus to find the right document.

Each tool is thin: it reads `settings.index_results_dir`, delegates to the
`pageindex.filesystem` / `agent.vlm` logic, and returns a JSON-friendly result.
"""

from __future__ import annotations

from langchain_core.tools import tool

from vlm_rag_index.agent.vlm import answer_with_images
from vlm_rag_index.core.config import settings
from vlm_rag_index.pageindex.filesystem import store, traversal, virtual_nodes
from vlm_rag_index.pageindex.llm.factory import get_default_client


@tool
async def list_documents() -> list[str]:
    """List the names of every indexed document available to query.

    Use this only when the question names no topic to search for; prefer
    `search_filesystem` for corpus-wide questions.
    """
    return store.list_documents(settings.index_results_dir)


@tool
async def get_document_structure(
    doc_name: str,
    node_id: str | None = None,
    max_depth: int | None = None,
) -> dict | str:
    """Return the heading outline of one document for navigation.

    Gives titles, page ranges (`start_index`/`end_index`), and summaries — but not
    page text. The outline is returned only a level or two deep to stay small; a
    node with a `subsection_count` has hidden subsections you can expand.

    Args:
        doc_name: The document to inspect.
        node_id: Pass a node's `node_id` to ZOOM IN on that section and reveal its
            subsections. Omit to get the top-level outline.
        max_depth: Force how many levels to return (default: an adaptive 1–2).

    Returns an error message if `doc_name` or `node_id` is unknown.
    """
    structure = store.document_structure(
        settings.index_results_dir, doc_name, node_id=node_id, max_depth=max_depth
    )
    if structure is None:
        return f"No indexed document named '{doc_name}'. Use list_documents to see options."
    return structure


@tool
async def answer_from_pages(doc_name: str, page_indices: list[int], question: str) -> str:
    """Read specific pages of a document with a vision model and answer the question.

    Pass a TIGHT 1-based page range (never more than ~20 pages), taken from
    `get_document_structure`. The model answers grounded in those page images and
    cites page numbers.
    """
    pdf = store.resolve_pdf(settings.index_results_dir, doc_name)
    if pdf is None:
        return f"No source PDF for '{doc_name}'. Use list_documents to see options."
    return await answer_with_images(pdf, page_indices, question, dpi=settings.vlm_dpi)


@tool
async def search_filesystem(query: str) -> list[str]:
    """Find documents relevant to a corpus-wide query.

    Navigates the corpus by document metadata (category, entities) and returns
    candidate document names ranked by relevance. START HERE for questions like
    "which document discusses X".
    """
    corpus = store.load_corpus(settings.index_results_dir)
    if not corpus:
        return []
    client = get_default_client(settings.filesystem_model)
    return await traversal.search_filesystem(
        query, corpus, client=client, settings=settings
    )


@tool
async def list_folder(path: str = "/") -> list[dict]:
    """Browse the corpus by category.

    `path="/"` lists the top-level folders; `path="/<folder>"` lists the
    documents filed under that folder.
    """
    corpus = store.load_corpus(settings.index_results_dir)
    return virtual_nodes.folder_children(corpus, path, settings.filesystem_axes_default)


@tool
async def get_folder_summary(path: str) -> str:
    """Describe what a corpus folder contains (its documents and their summaries)."""
    corpus = store.load_corpus(settings.index_results_dir)
    return virtual_nodes.folder_summary(corpus, path, settings.filesystem_axes_default)


TOOLS = [
    search_filesystem,
    list_folder,
    get_folder_summary,
    list_documents,
    get_document_structure,
    answer_from_pages,
]
