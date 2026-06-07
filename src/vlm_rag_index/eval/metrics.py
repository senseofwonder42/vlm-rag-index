"""Pure scoring for the page-accuracy eval.

`score_row` turns one `(ManifestRow, AgentRun)` pair into a flat dict (also the
per-row record written to the results file); `summarize` aggregates many of them.
No I/O, no LLM — kept deterministic so it can be unit-tested directly.
"""

from __future__ import annotations

from vlm_rag_index.eval.agent_runner import AgentRun
from vlm_rag_index.eval.manifest import ManifestRow


def score_row(row: ManifestRow, run: AgentRun) -> dict:
    """Score one query's retrieval against its gold doc and pages.

    `doc_hit` is true when any `answer_from_pages` call targeted the gold
    document. `page_hit` is true when a call on the gold document read at least
    one gold page (compared in 1-based space).

    Args:
        row: The manifest row holding the gold answers.
        run: The agent's run for this row.

    Returns:
        A JSON-serializable record with the verdict and supporting fields.
    """
    gold_doc = row.gold_doc
    gold_pages = set(row.gold_pages_1based)

    doc_hit = any(doc == gold_doc for doc, _ in run.calls)
    page_hit = any(
        doc == gold_doc and gold_pages.intersection(pages) for doc, pages in run.calls
    )

    return {
        "query_id": row.query_id,
        "domain": row.domain,
        "language": row.language,
        "query": row.query,
        "gold_doc": gold_doc,
        "gold_pages": sorted(gold_pages),
        "calls": [{"doc_name": doc, "page_indices": pages} for doc, pages in run.calls],
        "final_answer": run.final_answer,
        "doc_hit": doc_hit,
        "page_hit": page_hit,
        "error": run.error,
    }


def _rate(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def _breakdown(scored: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for s in scored:
        groups.setdefault(s[key], []).append(s)
    return {
        name: {
            "n": len(rows),
            "doc_hit_rate": _rate(sum(r["doc_hit"] for r in rows), len(rows)),
            "page_hit_rate": _rate(sum(r["page_hit"] for r in rows), len(rows)),
        }
        for name, rows in sorted(groups.items())
    }


def summarize(scored: list[dict]) -> dict:
    """Aggregate per-row scores into overall and per-group rates.

    Args:
        scored: Records produced by `score_row`.

    Returns:
        Overall counts and rates (`page_hit_rate_given_doc_hit` conditions on the
        agent having found the right document) plus `by_domain`/`by_language`
        breakdowns.
    """
    total = len(scored)
    doc_hits = sum(s["doc_hit"] for s in scored)
    page_hits = sum(s["page_hit"] for s in scored)
    errors = sum(1 for s in scored if s["error"])

    return {
        "n": total,
        "errors": errors,
        "doc_hit_rate": _rate(doc_hits, total),
        "page_hit_rate": _rate(page_hits, total),
        "page_hit_rate_given_doc_hit": _rate(page_hits, doc_hits),
        "by_domain": _breakdown(scored, "domain"),
        "by_language": _breakdown(scored, "language"),
    }
