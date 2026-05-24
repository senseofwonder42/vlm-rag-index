import base64
from io import BytesIO
from pathlib import Path

import fitz

from vlm_rag_index.pageindex.llm.protocol import ImageUrlPart

MAX_LONG_EDGE_PX = 1568


def render_pages(
    source: Path | str | BytesIO,
    indices: list[int],
    dpi: int = 144,
) -> dict[int, bytes]:
    """Render the given 1-based page indices to PNG bytes.

    The long edge is capped at MAX_LONG_EDGE_PX; providers downscale internally beyond
    that, so paying for the extra upload bandwidth gains nothing.
    """
    out: dict[int, bytes] = {}
    doc = _open(source)
    try:
        for i in indices:
            page = doc.load_page(i - 1)
            zoom = _zoom_for_page(page, dpi)
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out[i] = pix.tobytes("png")
    finally:
        doc.close()
    return out


def image_part(png: bytes) -> ImageUrlPart:
    b64 = base64.b64encode(png).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def _open(source: Path | str | BytesIO) -> fitz.Document:
    if isinstance(source, BytesIO):
        return fitz.open(stream=source.getvalue(), filetype="pdf")
    return fitz.open(Path(source))


def _zoom_for_page(page: fitz.Page, dpi: int) -> float:
    base_zoom = dpi / 72.0
    rect = page.rect
    long_edge_pt = max(rect.width, rect.height)
    rendered_long_edge_px = long_edge_pt * base_zoom
    if rendered_long_edge_px <= MAX_LONG_EDGE_PX:
        return base_zoom
    return MAX_LONG_EDGE_PX / long_edge_pt
