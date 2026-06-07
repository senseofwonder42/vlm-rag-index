"""Default dataset locations, shared by the eval entry points."""

from __future__ import annotations

from pathlib import Path

from vlm_rag_index.core.constants import PROJECT_ROOT

DATASET_DIR = PROJECT_ROOT / "dataset"
PDFS_DIR = DATASET_DIR / "pdfs"
MANIFEST = DATASET_DIR / "manifest.jsonl"
# Injection outputs (symlinked PDFs + .json + .meta.json) live in a sibling
# `index/` of `pdfs/`, derived as `<dataset>/index`.
INDEX_DIR = DATASET_DIR / "index"
EVAL_RESULTS = DATASET_DIR / "eval_results.jsonl"


def index_dir_for(pdfs_dir: Path) -> Path:
    """Return the index directory for a given PDFs directory (`<dataset>/index`)."""
    return pdfs_dir.parent / "index"

# Where `subsample` writes a smaller copy by default.
SAMPLE_DIR = PROJECT_ROOT / "dataset" / "sample"
