import asyncio
from io import BytesIO
from pathlib import Path

from loguru import logger

from vlm_rag_index.core.config import Settings
from vlm_rag_index.core.config import settings as _default_settings
from vlm_rag_index.pageindex.llm.factory import get_default_client
from vlm_rag_index.pageindex.llm.protocol import LLMClient, Message
from vlm_rag_index.pageindex.observability import observe
from vlm_rag_index.pageindex.pdf.reader import extract_text_per_page
from vlm_rag_index.pageindex.prompts import render
from vlm_rag_index.pageindex.structured_responses import DocMetadata, MetadataSidecar
from vlm_rag_index.pageindex.tree.builder import embed_text, write_node_ids
from vlm_rag_index.pageindex.tree.post_process import reorder_keys, strip_internal_fields
from vlm_rag_index.pageindex.tree.types import IndexResult, TreeNode
from vlm_rag_index.pageindex.vlm.extract import (
    _assign_end_indices,
    _build_tree,
    _dense_rank_map,
    _verify_heading_text,
    aextract,
)
from vlm_rag_index.pageindex.vlm.reconcile import _reconcile_records
from vlm_rag_index.pageindex.vlm.summaries import aleaf_summaries, aparent_summaries


def _doc_name(pdf: Path | str | BytesIO) -> str:
    if isinstance(pdf, BytesIO):
        return "document.pdf"
    return Path(pdf).name


async def _doc_description(client: LLMClient, doc_name: str, tree: list) -> str:
    sections = [(n["title"], n.get("summary") or "") for n in tree]
    prompt = render("doc_description.j2", doc_name=doc_name, sections=sections)
    messages: list[Message] = [{"role": "user", "content": prompt}]
    return (await client.acomplete(messages)).strip()


@observe()
async def agenerate_metadata(
    result: IndexResult,
    client: LLMClient | None = None,
) -> MetadataSidecar:
    """Generate the `.meta.json` sidecar from an index result's top-level sections.

    One structured LLM call per document, prompted with the top-level section
    titles and summaries. The corpus filesystem (Phase 4) consumes these sidecars;
    the v1 agent does not.
    """
    c = client or get_default_client()

    sections = [(n.title, n.summary or "") for n in result.structure]
    prompt = render("doc_metadata.j2", doc_name=result.doc_name, sections=sections)
    messages: list[Message] = [{"role": "user", "content": prompt}]
    metadata = await c.acomplete_structured(messages, DocMetadata)
    return MetadataSidecar.from_metadata(result.doc_name, metadata)


@observe()
async def apage_index(
    pdf: Path | str | BytesIO,
    client: LLMClient | None = None,
    settings: Settings | None = None,
) -> IndexResult:
    s = settings or _default_settings
    c = client or get_default_client()

    logger.info("indexing {}", _doc_name(pdf))

    records, page_count = await aextract(pdf, c, s)
    if s.vlm_window_overlap > 0:
        records = _reconcile_records(records)

    rank_map = _dense_rank_map(records)
    tree = _build_tree(records, rank_map, last_page=page_count)
    _assign_end_indices(tree, records, last_page=page_count)

    if s.vlm_verify_heading_text and not isinstance(pdf, BytesIO):
        pages_text = extract_text_per_page(pdf)
        _verify_heading_text(tree, pages_text)

    if s.index_add_node_id:
        write_node_ids(tree)

    if s.index_add_node_summary:
        await aleaf_summaries(tree, c, pdf, s)
        await aparent_summaries(tree, c)

    if s.index_add_node_text and not isinstance(pdf, BytesIO):
        pages_text = extract_text_per_page(pdf)
        embed_text(tree, pages_text)

    doc_description: str | None = None
    if s.index_add_doc_description:
        if not s.index_add_node_summary:
            raise ValueError(
                "index_add_doc_description requires index_add_node_summary=True"
            )
        doc_description = await _doc_description(c, _doc_name(pdf), tree)

    strip_internal_fields(tree)
    structure = [TreeNode.model_validate(node) for node in tree]
    return IndexResult(
        doc_name=_doc_name(pdf),
        structure=structure,
        doc_description=doc_description,
    )


def page_index(
    pdf: Path | str | BytesIO,
    client: LLMClient | None = None,
    settings: Settings | None = None,
) -> IndexResult:
    return asyncio.run(apage_index(pdf, client=client, settings=settings))


def to_dict(result: IndexResult) -> dict:
    """Serialise an IndexResult with canonical key order for stable JSON diffs."""
    payload = result.model_dump(exclude_none=True)
    payload["structure"] = [reorder_keys(n) for n in payload["structure"]]
    return payload
