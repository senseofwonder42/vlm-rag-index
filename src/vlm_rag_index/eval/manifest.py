"""Typed loader for the dataset's `manifest.jsonl`.

Each line is one query row. The two conventions that bite naive code live here:
`answer_page_index` is 0-based while the index/agent use 1-based pages, and the
`pdf` field is a path relative to `output/` whereas the indexed `doc_name` is the
bare filename.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ManifestRow(BaseModel):
    """One query/answer row from the manifest."""

    model_config = ConfigDict(extra="ignore")

    pdf: str
    query: str
    query_id: int
    answer: str
    answer_page_index: int | list[int]
    domain: str
    language: str
    relevance_score: int

    @property
    def gold_doc(self) -> str:
        """The indexed `doc_name` for this row (the PDF's bare filename)."""
        return Path(self.pdf).name

    @property
    def gold_pages_1based(self) -> list[int]:
        """Gold answer pages as 1-based indices (manifest stores them 0-based)."""
        raw = self.answer_page_index
        pages = raw if isinstance(raw, list) else [raw]
        return [p + 1 for p in pages]


def load_manifest(path: Path) -> list[ManifestRow]:
    """Load every non-empty line of `path` as a `ManifestRow`.

    Args:
        path: Path to the `manifest.jsonl` file.

    Returns:
        The parsed rows in file order.

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    with path.open(encoding="utf-8") as fh:
        return [ManifestRow.model_validate_json(line) for line in fh if line.strip()]
