"""Synthesize a navigable hierarchy from per-document metadata.

The on-disk layout is flat. These helpers project the corpus onto a chosen
metadata axis (a list-valued sidecar field) so the agent has folders to descend
into. A document with several values on an axis appears under each — the tree is
a projection of the metadata, not a fixed hierarchy.
"""

from __future__ import annotations

from vlm_rag_index.pageindex.filesystem.store import CorpusDoc

# Sidecar fields that are list-valued and therefore usable as folder axes.
AXIS_FIELDS: tuple[str, ...] = ("category", "entities", "keywords")


def _values(doc: CorpusDoc, axis: str) -> list[str]:
    return list(getattr(doc.metadata, axis, []) or [])


def available_axes(corpus: list[CorpusDoc], configured: list[str]) -> list[str]:
    """Configured axes that actually discriminate this corpus.

    An axis is useful only if it has at least two distinct values across the
    candidate documents; otherwise every document lands in one folder.
    """
    usable: list[str] = []
    for axis in configured:
        if axis not in AXIS_FIELDS:
            continue
        distinct = {v for doc in corpus for v in _values(doc, axis)}
        if len(distinct) >= 2:
            usable.append(axis)
    return usable


def project(corpus: list[CorpusDoc], axis: str) -> dict[str, list[str]]:
    """Bin documents by their values on `axis`: ``{label: [doc_name, ...]}``.

    Documents with no value on the axis are filed under ``"(unlabelled)"``.
    Labels are returned in sorted order for stable output.
    """
    bins: dict[str, list[str]] = {}
    for doc in corpus:
        labels = _values(doc, axis) or ["(unlabelled)"]
        for label in labels:
            bins.setdefault(label, [])
            if doc.doc_name not in bins[label]:
                bins[label].append(doc.doc_name)
    return {label: bins[label] for label in sorted(bins)}


def folder_children(
    corpus: list[CorpusDoc], path: str, axes: list[str]
) -> list[dict]:
    """List the children of a virtual folder for browsing.

    The corpus is projected on the first usable default axis. ``path="/"`` lists
    the axis values as folders; ``path="/<label>"`` lists the documents filed
    under that label.
    """
    usable = available_axes(corpus, axes)
    if not usable:
        return [{"name": doc.doc_name, "type": "document"} for doc in corpus]
    tree = project(corpus, usable[0])
    label = path.strip("/")
    if not label:
        return [
            {"name": name, "type": "folder", "count": len(docs)}
            for name, docs in tree.items()
        ]
    return [{"name": name, "type": "document"} for name in tree.get(label, [])]


def folder_summary(corpus: list[CorpusDoc], path: str, axes: list[str]) -> str:
    """A short, deterministic description of what a virtual folder contains."""
    usable = available_axes(corpus, axes)
    label = path.strip("/")
    if not usable or not label:
        return f"Corpus root: {len(corpus)} documents."
    names = set(project(corpus, usable[0]).get(label, []))
    docs = [d for d in corpus if d.doc_name in names]
    if not docs:
        return f"No documents under '{label}'."
    lines = [f"'{label}' ({usable[0]}): {len(docs)} documents."]
    lines += [f"- {d.doc_name}: {d.metadata.summary}" for d in docs]
    return "\n".join(lines)
