"""The agent's one reach into the pipeline's LLM stack: a VLM read of pages.

`answer_from_pages` renders the requested pages and asks the same vision model
the indexer uses to answer a question grounded in those images. This is free
text (`acomplete`), not structured output — the images *are* the context.
"""

from __future__ import annotations

from pathlib import Path

from vlm_rag_index.pageindex.llm.factory import get_default_client
from vlm_rag_index.pageindex.llm.protocol import ContentPart, LLMClient, Message
from vlm_rag_index.pageindex.observability import observe
from vlm_rag_index.pageindex.pdf.renderer import image_part, render_pages
from vlm_rag_index.pageindex.prompts import render


@observe()
async def answer_with_images(
    pdf: Path,
    page_indices: list[int],
    question: str,
    *,
    client: LLMClient | None = None,
    dpi: int = 144,
) -> str:
    """Render `page_indices` of `pdf` and answer `question` from the images.

    Args:
        pdf: Path to the source PDF.
        page_indices: 1-based page indices to read.
        question: The question to answer from those pages.
        client: LLM client to use; defaults to the pipeline's vision client.
        dpi: Render DPI for the page images.

    Returns:
        The model's grounded free-text answer.
    """
    c = client or get_default_client()
    rendered = render_pages(pdf, page_indices, dpi=dpi)
    ordered = sorted(rendered)
    text = render("vlm_answer.j2", question=question, page_indices=ordered)
    content: list[ContentPart] = [{"type": "text", "text": text}]
    content += [image_part(rendered[i]) for i in ordered]
    messages: list[Message] = [{"role": "user", "content": content}]
    return (await c.acomplete(messages)).strip()
