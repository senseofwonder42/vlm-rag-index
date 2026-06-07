"""Evaluate page-retrieval accuracy over the manifest (entry point: `eval-pages`).

For each query, runs the full agent pipeline (corpus search -> document -> pages),
recovers the pages it read, and scores them against the manifest's gold pages.
Writes a per-row JSONL and prints doc-hit / page-hit summaries.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from vlm_rag_index.eval._bootstrap import set_results_dir
from vlm_rag_index.eval.manifest import ManifestRow, load_manifest
from vlm_rag_index.eval.paths import EVAL_RESULTS, INDEX_DIR, MANIFEST

DEFAULT_MANIFEST = MANIFEST
DEFAULT_RESULTS_DIR = INDEX_DIR
DEFAULT_OUT = EVAL_RESULTS


def _select(rows: list[ManifestRow], domain: str | None, limit: int | None) -> list[ManifestRow]:
    if domain is not None:
        rows = [r for r in rows if r.domain == domain]
    if limit is not None:
        rows = rows[:limit]
    return rows


async def _evaluate(rows: list[ManifestRow], concurrency: int) -> list[dict]:
    # Lazy import: build_graph() reads the redirected INDEX_RESULTS_DIR.
    from vlm_rag_index.agent.graph import build_graph
    from vlm_rag_index.eval.agent_runner import run_query
    from vlm_rag_index.eval.metrics import score_row

    graph = build_graph()
    semaphore = asyncio.Semaphore(concurrency)
    scored: list[dict] = []

    with Progress() as progress:
        task = progress.add_task("evaluating", total=len(rows))

        async def _one(row: ManifestRow) -> dict:
            async with semaphore:
                run = await run_query(graph, row.query)
            progress.advance(task)
            return score_row(row, run)

        scored = await asyncio.gather(*(_one(r) for r in rows))
    return list(scored)


def _print_summary(console: Console, summary: dict) -> None:
    console.print(
        f"\n[bold]Overall[/bold]  n={summary['n']}  errors={summary['errors']}  "
        f"doc-hit={summary['doc_hit_rate']:.1%}  page-hit={summary['page_hit_rate']:.1%}  "
        f"page-hit|doc-hit={summary['page_hit_rate_given_doc_hit']:.1%}"
    )
    for title, key in (("By domain", "by_domain"), ("By language", "by_language")):
        table = Table(title=title)
        table.add_column(key.removeprefix("by_"))
        table.add_column("n", justify="right")
        table.add_column("doc-hit", justify="right")
        table.add_column("page-hit", justify="right")
        for name, stats in summary[key].items():
            table.add_row(
                name, str(stats["n"]), f"{stats['doc_hit_rate']:.1%}", f"{stats['page_hit_rate']:.1%}"
            )
        console.print(table)


def main(argv: list[str] | None = None) -> int:
    """Parse args, run the eval over the manifest, and report results."""
    parser = argparse.ArgumentParser(prog="eval-pages", description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="path to manifest.jsonl")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="indexed PDFs dir")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="per-row results JSONL output")
    parser.add_argument("--limit", type=int, help="evaluate only the first N rows")
    parser.add_argument("--domain", help="evaluate only rows in this domain")
    parser.add_argument("--concurrency", type=int, default=5, help="concurrent agent runs")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")

    set_results_dir(Path(args.results_dir))
    from vlm_rag_index.eval.metrics import summarize

    console = Console()
    rows = _select(load_manifest(manifest_path), args.domain, args.limit)
    if not rows:
        raise SystemExit("no rows to evaluate")

    scored = asyncio.run(_evaluate(rows, args.concurrency))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for record in scored:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    console.print(f"wrote {out_path}")

    _print_summary(console, summarize(scored))
    return 0


if __name__ == "__main__":
    sys.exit(main())
