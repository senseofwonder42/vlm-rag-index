"""Locate and load the per-document `{pdf, json, meta.json}` triples on disk.

The `meta.json` sidecars *are* the corpus index — there is no global database.
This module is the single place that maps a `doc_name` to its files and reads
them. The v1 single-document tools need only `{pdf, json}`; the v2 filesystem
layer additionally needs the `meta.json` sidecar.

A `doc_name` is the source filename, e.g. ``annual_report_2023.pdf``. On disk the
triple shares a stem: ``<stem>.pdf``, ``<stem>.json``, ``<stem>.meta.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from vlm_rag_index.pageindex.structured_responses import MetadataSidecar
from vlm_rag_index.pageindex.tree.types import IndexResult, TreeNode

_META_SUFFIX = ".meta.json"


@dataclass(frozen=True)
class CorpusDoc:
    """A document with all three files present and its sidecar loaded."""

    doc_name: str
    pdf: Path
    index_path: Path
    meta_path: Path
    metadata: MetadataSidecar


def _stem_of(doc_name: str) -> str:
    return doc_name[:-4] if doc_name.endswith(".pdf") else doc_name


def list_documents(results_dir: Path) -> list[str]:
    """Return the `doc_name` of every index JSON that has a matching PDF.

    Documents whose source PDF is missing are skipped — `answer_from_pages`
    would fail on them anyway.
    """
    names: list[str] = []
    for index_path in sorted(results_dir.glob("*.json")):
        if index_path.name.endswith(_META_SUFFIX):
            continue
        if not index_path.with_suffix(".pdf").is_file():
            continue
        try:
            doc_name = json.loads(index_path.read_text(encoding="utf-8"))["doc_name"]
        except (OSError, json.JSONDecodeError, KeyError):
            logger.warning("skipping unreadable index {}", index_path)
            continue
        names.append(doc_name)
    return names


def load_index(results_dir: Path, doc_name: str) -> IndexResult | None:
    """Load the `IndexResult` for `doc_name`, or None if it does not exist."""
    index_path = results_dir / f"{_stem_of(doc_name)}.json"
    if not index_path.is_file():
        return None
    return IndexResult.model_validate_json(index_path.read_text(encoding="utf-8"))


def resolve_pdf(results_dir: Path, doc_name: str) -> Path | None:
    """Return the path to `doc_name`'s source PDF, or None if missing."""
    pdf = results_dir / f"{_stem_of(doc_name)}.pdf"
    return pdf if pdf.is_file() else None


# When no explicit depth is asked for, expand the outline level by level while
# the number of included nodes stays under this budget. Keeps the default tool
# result small for deep documents (return level 1) but rich for shallow ones
# (return level 2), without ever dumping a whole 200-page tree into the agent.
_AUTO_NODE_BUDGET = 50


def document_structure(
    results_dir: Path,
    doc_name: str,
    *,
    node_id: str | None = None,
    max_depth: int | None = None,
) -> dict | str | None:
    """Return a depth-limited view of `doc_name`'s heading tree for navigation.

    The agent navigates on titles, page ranges, and summaries; the embedded page
    `text` is dropped, and the tree is truncated so deep documents don't flood the
    agent's context. Truncated nodes carry a `subsection_count` hint, and the agent
    can expand any of them by passing its `node_id`.

    Args:
        results_dir: Directory holding the index JSONs.
        doc_name: The document to read.
        node_id: When given, root the view at this node and expand its children
            (i.e. zoom in). When None, view the document's top-level sections.
        max_depth: Number of levels of the returned child list to include. When
            None, an adaptive depth (1–2, by node count) is chosen.

    Returns:
        The trimmed tree as a dict, a short message string when `node_id` is not
        found, or None when the document itself does not exist.
    """
    result = load_index(results_dir, doc_name)
    if result is None:
        return None

    if node_id is not None:
        node = _find_node(result.structure, node_id)
        if node is None:
            return f"No node '{node_id}' in '{doc_name}'. Call without node_id to see the outline."
        levels = max_depth if max_depth is not None else _auto_levels(node.nodes)
        view = _node_fields(node)
        view["doc_name"] = result.doc_name
        if node.nodes:
            view["nodes"] = [_shape(child, levels) for child in node.nodes]
        return view

    levels = max_depth if max_depth is not None else _auto_levels(result.structure)
    return {
        "doc_name": result.doc_name,
        "doc_description": result.doc_description,
        "structure": [_shape(n, levels) for n in result.structure],
    }


def _node_fields(node: TreeNode) -> dict:
    out: dict = {
        "node_id": node.node_id,
        "title": node.title,
        "start_index": node.start_index,
        "end_index": node.end_index,
    }
    if node.summary is not None:
        out["summary"] = node.summary
    return out


def _shape(node: TreeNode, levels: int) -> dict:
    """Strip `text` and keep only `levels` levels of descendants below `node`.

    A node whose children are cut off by the depth limit gets a `subsection_count`
    so the caller knows it can be expanded via `node_id`.
    """
    out = _node_fields(node)
    if node.nodes:
        if levels > 1:
            out["nodes"] = [_shape(child, levels - 1) for child in node.nodes]
        else:
            out["subsection_count"] = len(node.nodes)
    return out


def _count_to_depth(nodes: list[TreeNode], depth: int) -> int:
    if depth <= 0:
        return 0
    return sum(1 + _count_to_depth(n.nodes, depth - 1) for n in nodes)


def _auto_levels(nodes: list[TreeNode]) -> int:
    """Deepest level count whose included-node total still fits the budget (min 1)."""
    levels = 1
    while True:
        deeper = _count_to_depth(nodes, levels + 1)
        if deeper > _AUTO_NODE_BUDGET or deeper == _count_to_depth(nodes, levels):
            return levels
        levels += 1


def _find_node(nodes: list[TreeNode], node_id: str) -> TreeNode | None:
    for node in nodes:
        if node.node_id == node_id:
            return node
        found = _find_node(node.nodes, node_id)
        if found is not None:
            return found
    return None


def load_corpus(results_dir: Path) -> list[CorpusDoc]:
    """Load every document that has a complete `{pdf, json, meta.json}` triple.

    Documents missing any of the three files are skipped: without the sidecar
    they are invisible to the corpus filesystem, and without the PDF they cannot
    be answered from.
    """
    corpus: list[CorpusDoc] = []
    for meta_path in sorted(results_dir.glob(f"*{_META_SUFFIX}")):
        stem = meta_path.name[: -len(_META_SUFFIX)]
        pdf = results_dir / f"{stem}.pdf"
        index_path = results_dir / f"{stem}.json"
        if not (pdf.is_file() and index_path.is_file()):
            continue
        try:
            metadata = MetadataSidecar.model_validate_json(
                meta_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            logger.warning("skipping unreadable sidecar {}", meta_path)
            continue
        corpus.append(
            CorpusDoc(
                doc_name=metadata.doc_name,
                pdf=pdf,
                index_path=index_path,
                meta_path=meta_path,
                metadata=metadata,
            )
        )
    return corpus
