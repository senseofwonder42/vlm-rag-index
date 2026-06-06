"""The v1 single-document tools, invoked through their LangChain wrappers."""

import json
from pathlib import Path

import pytest

from vlm_rag_index.agent import tools
from vlm_rag_index.core.config import settings


def _write_doc(results_dir: Path, stem: str, structure: list[dict], pdf: bool = True) -> str:
    doc_name = f"{stem}.pdf"
    if pdf:
        (results_dir / doc_name).write_bytes(b"%PDF-1.4 fake\n")
    (results_dir / f"{stem}.json").write_text(
        json.dumps({"doc_name": doc_name, "structure": structure}), encoding="utf-8"
    )
    return doc_name


@pytest.fixture
def docs_dir(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(settings, "index_results_dir", tmp_path)
    return tmp_path


async def test_list_documents(docs_dir: Path):
    _write_doc(docs_dir, "a", [{"title": "x", "start_index": 1, "end_index": 1}])
    _write_doc(docs_dir, "b", [{"title": "y", "start_index": 1, "end_index": 1}], pdf=False)

    result = await tools.list_documents.ainvoke({})

    assert result == ["a.pdf"]  # b has no PDF, skipped


async def test_get_document_structure_strips_text(docs_dir: Path):
    _write_doc(
        docs_dir,
        "a",
        [{"title": "Intro", "start_index": 1, "end_index": 2, "text": "SECRET TEXT"}],
    )

    result = await tools.get_document_structure.ainvoke({"doc_name": "a.pdf"})

    assert isinstance(result, dict)
    assert "SECRET TEXT" not in str(result)
    assert result["structure"][0]["title"] == "Intro"


async def test_get_document_structure_zoom_by_node_id(docs_dir: Path):
    structure = [
        {
            "title": "1 Methods",
            "node_id": "0001",
            "start_index": 1,
            "end_index": 9,
            "nodes": [
                {"title": "1.1 Setup", "node_id": "0002", "start_index": 1, "end_index": 4}
            ],
        }
    ]
    _write_doc(docs_dir, "a", structure)

    outline = await tools.get_document_structure.ainvoke({"doc_name": "a.pdf", "max_depth": 1})
    assert outline["structure"][0]["subsection_count"] == 1

    zoomed = await tools.get_document_structure.ainvoke(
        {"doc_name": "a.pdf", "node_id": "0001"}
    )
    assert zoomed["node_id"] == "0001"
    assert zoomed["nodes"][0]["title"] == "1.1 Setup"


async def test_get_document_structure_unknown_returns_message(docs_dir: Path):
    result = await tools.get_document_structure.ainvoke({"doc_name": "missing.pdf"})
    assert isinstance(result, str)
    assert "missing.pdf" in result


async def test_answer_from_pages_resolves_pdf_and_delegates(docs_dir: Path, monkeypatch):
    _write_doc(docs_dir, "a", [{"title": "x", "start_index": 1, "end_index": 3}])
    captured = {}

    async def fake_answer(pdf, page_indices, question, **kwargs):  # noqa: ARG001
        captured["pdf"] = pdf
        captured["pages"] = page_indices
        return "answer (page 2)"

    monkeypatch.setattr(tools, "answer_with_images", fake_answer)

    result = await tools.answer_from_pages.ainvoke(
        {"doc_name": "a.pdf", "page_indices": [2], "question": "what?"}
    )

    assert result == "answer (page 2)"
    assert captured["pdf"].name == "a.pdf"
    assert captured["pages"] == [2]


async def test_answer_from_pages_missing_pdf(docs_dir: Path):
    result = await tools.answer_from_pages.ainvoke(
        {"doc_name": "nope.pdf", "page_indices": [1], "question": "q"}
    )
    assert isinstance(result, str)
    assert "nope.pdf" in result
