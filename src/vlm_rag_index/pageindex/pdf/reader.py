from pathlib import Path

import fitz


def count_pages(pdf: Path | str) -> int:
    with fitz.open(Path(pdf)) as doc:
        return doc.page_count


def extract_text_per_page(pdf: Path | str) -> dict[int, str]:
    """Return plain-text content per page, 1-based page indices."""
    out: dict[int, str] = {}
    with fitz.open(Path(pdf)) as doc:
        for i, page in enumerate(doc, start=1):
            out[i] = page.get_text("text")
    return out
