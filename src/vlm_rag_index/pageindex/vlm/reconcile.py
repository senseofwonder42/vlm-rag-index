import re
from collections import Counter
from difflib import SequenceMatcher

from vlm_rag_index.pageindex.structured_responses import PageHeading, PageInfo, VisualRank

_NORM_LEADING_NUMBERING = re.compile(r"^\s*(?:[a-z]\.\d+|\d+(?:\.\d+)*)\s+", re.IGNORECASE)
_NORM_PUNCT = re.compile(r"[^\w\s]")
_NORM_WS = re.compile(r"\s+")
_CLUSTER_THRESHOLD = 0.92


def _norm(text: str) -> str:
    t = text.strip().lower()
    t = _NORM_LEADING_NUMBERING.sub("", t)
    t = _NORM_PUNCT.sub(" ", t)
    t = _NORM_WS.sub(" ", t).strip()
    return t


def _reconcile_records(records: list[PageInfo]) -> list[PageInfo]:
    if not records:
        return []

    # Fast-path: every page in exactly one record -> no reconciliation needed.
    page_counts = Counter(r.page_index for r in records)
    if all(c == 1 for c in page_counts.values()):
        return list(records)

    # Cluster headings by normalised similarity (union-find).
    sources: list[tuple[int, int, int, PageHeading]] = []  # (record_idx, heading_idx, page, heading)
    for ri, rec in enumerate(records):
        for hi, h in enumerate(rec.headings):
            sources.append((ri, hi, rec.page_index, h))

    parent = list(range(len(sources)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    norms = [_norm(s[3].text) for s in sources]
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            if not norms[i] or not norms[j]:
                continue
            if SequenceMatcher(None, norms[i], norms[j]).ratio() >= _CLUSTER_THRESHOLD:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(len(sources)):
        clusters.setdefault(find(i), []).append(i)

    # For each cluster decide canonical (page, text, rank).
    canonical: list[tuple[int, str, VisualRank]] = []
    for members in clusters.values():
        pages = [sources[m][2] for m in members]
        page_mode = Counter(pages).most_common()
        # Sort: highest count desc, then smaller page asc.
        page_mode.sort(key=lambda x: (-x[1], x[0]))
        canonical_page = page_mode[0][0]
        canonical_text = max((sources[m][3].text for m in members), key=len)
        ranks = [sources[m][3].visual_rank for m in members]
        rank_mode = Counter(ranks).most_common(1)[0][0]
        canonical.append((canonical_page, canonical_text, rank_mode))

    # Group canonical headings by page.
    headings_per_page: dict[int, list[PageHeading]] = {}
    for page, text, rank in canonical:
        headings_per_page.setdefault(page, []).append(
            PageHeading(text=text, visual_rank=rank)
        )

    ending_sets: dict[int, list[str]] = {}
    for rec in records:
        ending_sets.setdefault(rec.page_index, []).extend(rec.ending_sections)

    out: list[PageInfo] = []
    for page in sorted(set(r.page_index for r in records)):
        seen_norms: set[str] = set()
        merged_endings: list[str] = []
        for raw in ending_sets.get(page, []):
            key = _norm(raw)
            if key in seen_norms:
                continue
            seen_norms.add(key)
            merged_endings.append(raw)
        out.append(
            PageInfo(
                page_index=page,
                headings=headings_per_page.get(page, []),
                ending_sections=merged_endings,
            )
        )
    return out
