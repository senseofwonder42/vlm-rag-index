import re
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from loguru import logger

from vlm_rag_index.core.config import Settings
from vlm_rag_index.core.config import settings as _default_settings
from vlm_rag_index.pageindex.llm.protocol import ContentPart, LLMClient, Message, TextPart
from vlm_rag_index.pageindex.pdf.reader import count_pages
from vlm_rag_index.pageindex.pdf.renderer import image_part, render_pages
from vlm_rag_index.pageindex.prompts import render
from vlm_rag_index.pageindex.structured_responses import (
    PageBatchResponse,
    PageInfo,
    VisualRank,
)

_VISUAL_RANK_ORDER: tuple[VisualRank, ...] = ("xlarge", "large", "medium", "small", "xsmall")
_CAPTION_RE = re.compile(
    r"^(table|figure|fig\.|algorithm|alg\.|equation|eq\.|listing|scheme)\s*\d",
    re.IGNORECASE,
)
_NUMBERING_RE = re.compile(r"^\s*(?:[A-Z]\.\d+|\d+(?:\.\d+)*)")


def _sliding_windows(seq: list[int], size: int, overlap: int) -> list[list[int]]:
    if size <= 0:
        raise ValueError("size must be positive")
    if not seq:
        return []
    overlap = max(0, min(overlap, size - 1))
    step = size - overlap
    windows: list[list[int]] = []
    i = 0
    while i < len(seq):
        windows.append(seq[i : i + size])
        if i + size >= len(seq):
            break
        i += step
    return windows


def _depth_from_numbering(text: str) -> int | None:
    m = _NUMBERING_RE.match(text)
    if not m:
        return None
    token = m.group(0).strip()
    # "1" -> depth 1, "1.2" -> depth 2, "A.1" -> depth 2.
    return token.count(".") + 1


def _dense_rank_map(records: list[PageInfo]) -> dict[VisualRank, int]:
    seen: set[VisualRank] = set()
    for r in records:
        for h in r.headings:
            seen.add(h.visual_rank)
    ordered = [rank for rank in _VISUAL_RANK_ORDER if rank in seen]
    return {rank: depth for depth, rank in enumerate(ordered, start=1)}


def _is_caption(text: str) -> bool:
    return bool(_CAPTION_RE.match(text.strip()))


async def _call_batch(
    client: LLMClient,
    window: list[int],
    rendered: dict[int, bytes],
) -> list[PageInfo]:
    prompt_text = render(
        "vlm_page_batch.j2", num_pages=len(window), page_indices=window
    )
    content: list[ContentPart] = [cast(TextPart, {"type": "text", "text": prompt_text})]
    for p in window:
        content.append(image_part(rendered[p]))
    messages: list[Message] = [{"role": "user", "content": content}]
    response = await client.acomplete_structured(messages, PageBatchResponse)
    out: list[PageInfo] = []
    window_set = set(window)
    dropped = 0
    for rec in response.pages:
        if rec.page_index in window_set:
            out.append(rec)
        else:
            dropped += 1
    if dropped:
        logger.warning(
            "VLM batch returned {} record(s) outside requested window {}", dropped, window
        )
    return out


async def aextract(
    pdf: Path | str | BytesIO,
    client: LLMClient,
    settings: Settings | None = None,
) -> tuple[list[PageInfo], int]:
    s = settings or _default_settings
    page_count = count_pages(pdf) if not isinstance(pdf, BytesIO) else _count_bytesio(pdf)
    indices = list(range(1, page_count + 1))
    size = min(s.vlm_pages_per_batch, s.vlm_max_images_per_call)
    windows = _sliding_windows(indices, size=size, overlap=s.vlm_window_overlap)
    records: list[PageInfo] = []
    for window in windows:
        rendered = render_pages(pdf, window, dpi=s.vlm_dpi)
        records.extend(await _call_batch(client, window, rendered))
    return records, page_count


def _count_bytesio(buf: BytesIO) -> int:
    import fitz

    with fitz.open(stream=buf.getvalue(), filetype="pdf") as doc:
        return doc.page_count


def _build_tree(
    records: list[PageInfo],
    rank_to_depth: dict[VisualRank, int],
    last_page: int,
) -> list[dict[str, Any]]:
    """Walk records in page order, push/pop a parent stack, return nested dicts."""
    by_page = {r.page_index: r for r in records}
    ordered_pages = sorted(by_page)
    if not ordered_pages:
        return []

    nodes_in_order: list[dict[str, Any]] = []
    for page in ordered_pages:
        rec = by_page[page]
        for h in rec.headings:
            if _is_caption(h.text):
                continue
            depth_num = _depth_from_numbering(h.text)
            depth = depth_num if depth_num is not None else rank_to_depth.get(h.visual_rank)
            if depth is None or depth < 1:
                continue
            nodes_in_order.append(
                {
                    "title": h.text.strip(),
                    "start_index": page,
                    "end_index": None,
                    "nodes": [],
                    "_depth": depth,
                }
            )

    if not nodes_in_order:
        # No headings detected anywhere — single leaf spanning the whole doc.
        leaf: dict[str, Any] = {
            "title": "Document",
            "start_index": 1,
            "end_index": last_page,
            "nodes": [],
            "_depth": 1,
        }
        return [leaf]

    # Synthetic Front Matter for content before the first heading.
    first_heading_page = nodes_in_order[0]["start_index"]
    roots: list[dict[str, Any]] = []
    if first_heading_page > 1:
        roots.append(
            {
                "title": "Front Matter",
                "start_index": 1,
                "end_index": first_heading_page - 1,
                "nodes": [],
                "_depth": 1,
            }
        )

    stack: list[dict[str, Any]] = []  # nodes currently open (depth-ordered)
    for node in nodes_in_order:
        while stack and stack[-1]["_depth"] >= node["_depth"]:
            stack.pop()
        if stack:
            stack[-1]["nodes"].append(node)
        else:
            roots.append(node)
        stack.append(node)

    return roots


def _norm_title(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"^(?:[a-z]\.\d+|\d+(?:\.\d+)*)\s+", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _verify_heading_text(
    tree: list[dict[str, Any]],
    pages_text: dict[int, str],
    threshold: float = 0.8,
) -> None:
    """Snap each node's `title` to the closest text-layer line on its start page.

    Replaces a VLM paraphrase with the verbatim heading text when a close match
    exists. Front Matter / Document fallback nodes are skipped.
    """

    def walk(nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            if node["title"] in ("Front Matter", "Document"):
                walk(node["nodes"])
                continue
            page_text = pages_text.get(node["start_index"], "")
            best: tuple[float, str] = (0.0, node["title"])
            for line in page_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                ratio = SequenceMatcher(None, line.lower(), node["title"].lower()).ratio()
                if ratio > best[0]:
                    best = (ratio, line)
            if best[0] >= threshold:
                node["title"] = best[1]
            walk(node["nodes"])

    walk(tree)


def _assign_end_indices(
    tree: list[dict[str, Any]],
    records: list[PageInfo],
    last_page: int,
) -> None:
    """Two-pass: VLM `ending_sections` signal first, sibling-handoff fallback second."""
    # Collect (page, normalised_title) -> latest page across all records.
    ending_signals: list[tuple[int, str]] = []
    for rec in records:
        for raw in rec.ending_sections:
            ending_signals.append((rec.page_index, _norm_title(raw)))

    nodes_flat: list[dict[str, Any]] = []

    def collect(n: dict[str, Any]) -> None:
        nodes_flat.append(n)
        for c in n["nodes"]:
            collect(c)

    for r in tree:
        collect(r)

    # Pass 1: VLM signal.
    for node in nodes_flat:
        if node["title"] == "Front Matter":
            continue  # Front Matter's end is set at build time.
        if node["title"] == "Document":
            continue  # No-heading fallback already spans full doc.
        node_norm = _norm_title(node["title"])
        latest: int | None = None
        for page, ending in ending_signals:
            if page < node["start_index"]:
                continue
            if SequenceMatcher(None, ending, node_norm).ratio() >= 0.85:
                if latest is None or page > latest:
                    latest = page
        if latest is not None:
            node["end_index"] = max(latest, node["start_index"])

    # Pass 2: sibling-handoff fallback. Walk siblings; the next sibling's start_index - 0
    # serves as this node's end_index. (We intentionally use the next sibling's
    # start_index so adjacent sections meet at a page boundary the agent can reason
    # about; documents do not split mid-page in this index.)
    def fill(siblings: list[dict[str, Any]], parent_end: int) -> None:
        for i, n in enumerate(siblings):
            if n["end_index"] is None:
                if i + 1 < len(siblings):
                    n["end_index"] = max(siblings[i + 1]["start_index"], n["start_index"])
                else:
                    n["end_index"] = max(parent_end, n["start_index"])
            fill(n["nodes"], n["end_index"])

    fill(tree, last_page)
