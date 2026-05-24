from io import BytesIO
from pathlib import Path
from typing import Any, cast

from vlm_rag_index.core.config import Settings
from vlm_rag_index.pageindex.llm.protocol import ContentPart, LLMClient, Message, TextPart
from vlm_rag_index.pageindex.pdf.renderer import image_part, render_pages
from vlm_rag_index.pageindex.prompts import render


def _chunk_pages(start: int, end: int, size: int) -> list[list[int]]:
    pages = list(range(start, end + 1))
    return [pages[i : i + size] for i in range(0, len(pages), size)]


async def _leaf_chunk_summary(
    client: LLMClient,
    pdf: Path | str | BytesIO,
    node: dict[str, Any],
    chunk: list[int],
    dpi: int,
    multi_chunk: bool,
) -> str:
    rendered = render_pages(pdf, chunk, dpi=dpi)
    prompt_text = render(
        "leaf_summary_chunk.j2",
        title=node["title"],
        start_index=node["start_index"],
        end_index=node["end_index"],
        chunk_pages=f"{chunk[0]}–{chunk[-1]}" if len(chunk) > 1 else str(chunk[0]),
        multi_chunk=multi_chunk,
    )
    content: list[ContentPart] = [cast(TextPart, {"type": "text", "text": prompt_text})]
    for p in chunk:
        content.append(image_part(rendered[p]))
    messages: list[Message] = [{"role": "user", "content": content}]
    return (await client.acomplete(messages)).strip()


async def _merge_leaf_partials(
    client: LLMClient,
    node: dict[str, Any],
    partials: list[tuple[str, str]],
) -> str:
    prompt = render(
        "leaf_summary_merge.j2",
        title=node["title"],
        start_index=node["start_index"],
        end_index=node["end_index"],
        partials=partials,
    )
    messages: list[Message] = [{"role": "user", "content": prompt}]
    return (await client.acomplete(messages)).strip()


async def _leaf_summary(
    client: LLMClient,
    pdf: Path | str | BytesIO,
    node: dict[str, Any],
    settings: Settings,
) -> str:
    size = settings.vlm_max_images_per_call
    chunks = _chunk_pages(node["start_index"], node["end_index"], size)
    multi = len(chunks) > 1
    partials: list[tuple[str, str]] = []
    for chunk in chunks:
        summary = await _leaf_chunk_summary(client, pdf, node, chunk, settings.vlm_dpi, multi)
        label = f"{chunk[0]}–{chunk[-1]}" if len(chunk) > 1 else str(chunk[0])
        partials.append((label, summary))
    if len(partials) == 1:
        return partials[0][1]
    return await _merge_leaf_partials(client, node, partials)


async def _parent_summary(client: LLMClient, node: dict[str, Any]) -> str:
    children = [(c["title"], c.get("summary") or "") for c in node["nodes"]]
    prompt = render(
        "parent_summary.j2",
        title=node["title"],
        start_index=node["start_index"],
        end_index=node["end_index"],
        children=children,
    )
    messages: list[Message] = [{"role": "user", "content": prompt}]
    return (await client.acomplete(messages)).strip()


async def aleaf_summaries(
    tree: list[dict[str, Any]],
    client: LLMClient,
    pdf: Path | str | BytesIO,
    settings: Settings,
) -> None:
    """Walk leaves depth-first and attach `summary` strings in place.

    Each leaf's pages are chunked by `settings.vlm_max_images_per_call`; one VLM
    call per chunk produces a partial summary, then a text LLM call merges
    multiple partials. Single-chunk leaves skip the merge.
    """

    async def walk(nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            if node["nodes"]:
                await walk(node["nodes"])
            else:
                node["summary"] = await _leaf_summary(client, pdf, node, settings)

    await walk(tree)


async def aparent_summaries(tree: list[dict[str, Any]], client: LLMClient) -> None:
    """Bottom-up walk: every non-leaf gets a summary derived from its children."""

    async def walk(nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            if node["nodes"]:
                await walk(node["nodes"])
                node["summary"] = await _parent_summary(client, node)

    await walk(tree)
