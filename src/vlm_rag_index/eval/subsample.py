"""Build a smaller copy of the dataset for quick test runs (entry point: `subsample`).

Selects a few PDFs (by default the first N per domain), copies them into
`<dest>/pdfs/`, and writes a `<dest>/manifest.jsonl` containing only the rows
that reference them. Manifest lines are copied field-for-field (not re-parsed),
so nothing is lost. Point `inject` / `eval-pages` at `<dest>` to test on the
subset.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from loguru import logger

from vlm_rag_index.eval.paths import MANIFEST, PDFS_DIR, SAMPLE_DIR


def _select_pdfs(
    rows: list[dict], per_domain: int, domains: set[str] | None
) -> list[str]:
    """Pick up to `per_domain` distinct PDFs per domain, in manifest order."""
    chosen: dict[str, list[str]] = {}
    for row in rows:
        domain = row["domain"]
        if domains is not None and domain not in domains:
            continue
        bucket = chosen.setdefault(domain, [])
        pdf = row["pdf"]
        if pdf not in bucket and len(bucket) < per_domain:
            bucket.append(pdf)
    return [pdf for bucket in chosen.values() for pdf in bucket]


def main(argv: list[str] | None = None) -> int:
    """Create a subsampled dataset under `--dest`."""
    parser = argparse.ArgumentParser(prog="subsample", description=__doc__)
    parser.add_argument("--manifest", default=str(MANIFEST), help="source manifest.jsonl")
    parser.add_argument("--pdfs-dir", default=str(PDFS_DIR), help="source PDFs directory")
    parser.add_argument("--dest", default=str(SAMPLE_DIR), help="output dataset directory")
    parser.add_argument(
        "--per-domain", type=int, default=1, help="PDFs to keep per domain (default: 1)"
    )
    parser.add_argument(
        "--domains", help="comma-separated domains to include (default: all)"
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    pdfs_dir = Path(args.pdfs_dir)
    dest = Path(args.dest)
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")

    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    domains = {d.strip() for d in args.domains.split(",")} if args.domains else None
    keep = set(_select_pdfs(rows, args.per_domain, domains))
    if not keep:
        raise SystemExit("no PDFs matched the selection")

    dest_pdfs = dest / "pdfs"
    dest_pdfs.mkdir(parents=True, exist_ok=True)
    for pdf_rel in sorted(keep):
        src = pdfs_dir / Path(pdf_rel).name
        if not src.is_file():
            raise SystemExit(f"source PDF missing: {src}")
        shutil.copy2(src, dest_pdfs / src.name)

    kept_rows = [r for r in rows if r["pdf"] in keep]
    out_manifest = dest / "manifest.jsonl"
    with out_manifest.open("w", encoding="utf-8") as fh:
        for row in kept_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    index_dir = dest / "index"
    logger.info(
        "subsampled {} PDFs / {} rows -> {}", len(keep), len(kept_rows), dest
    )
    logger.info("next: uv run inject --pdfs-dir {}", dest_pdfs)
    logger.info(
        "then: uv run eval-pages --manifest {} --results-dir {}", out_manifest, index_dir
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
