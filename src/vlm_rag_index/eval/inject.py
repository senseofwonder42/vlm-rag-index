"""Inject the dataset corpus into the index (entry point: `inject`).

The agent's store resolves a document's `.pdf`, `.json`, and `.meta.json` by a
shared stem within one directory, so all three must live together. To keep the
source `pdfs/` directory pristine, `inject` symlinks each PDF into a sibling
`<dataset>/index/` directory and writes the index JSON + metadata sidecar there,
next to the symlink. Point `ask` / `eval-pages` at that `index/` directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from vlm_rag_index.cli.index import main as index_main
from vlm_rag_index.eval.paths import PDFS_DIR, index_dir_for


def link_pdfs(pdfs_dir: Path, index_dir: Path) -> int:
    """Symlink every `*.pdf` in `pdfs_dir` into `index_dir` (idempotent).

    Args:
        pdfs_dir: Directory of source PDFs.
        index_dir: Directory to populate with symlinks (created if missing).

    Returns:
        The number of PDFs linked (existing links are reused, not recreated).
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(pdfs_dir.glob("*.pdf"))
    for pdf in pdfs:
        link = index_dir / pdf.name
        if not link.is_symlink() and not link.exists():
            link.symlink_to(pdf.resolve())
    return len(pdfs)


def main(argv: list[str] | None = None) -> int:
    """Symlink the dataset PDFs into the index dir and run the indexer there."""
    parser = argparse.ArgumentParser(prog="inject", description=__doc__)
    parser.add_argument(
        "--pdfs-dir",
        default=str(PDFS_DIR),
        help=f"directory of source PDFs to index (default: {PDFS_DIR})",
    )
    parser.add_argument(
        "--index-dir",
        help="where to write symlinks + index JSONs (default: <dataset>/index)",
    )
    args = parser.parse_args(argv)

    pdfs_dir = Path(args.pdfs_dir)
    if not pdfs_dir.is_dir():
        raise SystemExit(f"--pdfs-dir not found: {pdfs_dir}")
    index_dir = Path(args.index_dir) if args.index_dir else index_dir_for(pdfs_dir)

    count = link_pdfs(pdfs_dir, index_dir)
    logger.info("linked {} PDFs into {}", count, index_dir)
    return index_main(["--input-dir", str(index_dir)])


if __name__ == "__main__":
    sys.exit(main())
