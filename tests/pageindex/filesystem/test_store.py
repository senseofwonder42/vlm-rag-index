from pathlib import Path

from vlm_rag_index.pageindex.filesystem import store

from .conftest import write_doc


def test_load_corpus_finds_complete_triples(corpus_dir: Path):
    corpus = store.load_corpus(corpus_dir)
    assert len(corpus) == 21
    assert {d.doc_name for d in corpus} >= {"fin_q3_2023.pdf", "ml_rlhf.pdf"}


def test_load_corpus_skips_incomplete_triples(tmp_path: Path):
    write_doc(tmp_path, "complete", category=["a"], summary="ok")
    write_doc(tmp_path, "no_pdf", category=["a"], summary="x", pdf=False)
    write_doc(tmp_path, "no_meta", category=["a"], summary="x", meta=False)

    corpus = store.load_corpus(tmp_path)

    assert [d.doc_name for d in corpus] == ["complete.pdf"]


def test_list_documents_requires_matching_pdf(tmp_path: Path):
    write_doc(tmp_path, "has_pdf", category=["a"], summary="x")
    write_doc(tmp_path, "no_pdf", category=["a"], summary="x", pdf=False)

    assert store.list_documents(tmp_path) == ["has_pdf.pdf"]


def test_document_structure_strips_text(tmp_path: Path):
    structure = [
        {
            "title": "1 Intro",
            "node_id": "0001",
            "start_index": 1,
            "end_index": 3,
            "summary": "intro",
            "text": "HUGE PAGE TEXT",
            "nodes": [
                {
                    "title": "1.1 Sub",
                    "start_index": 1,
                    "end_index": 2,
                    "text": "more text",
                }
            ],
        }
    ]
    write_doc(tmp_path, "doc", category=["a"], summary="x", structure=structure)

    out = store.document_structure(tmp_path, "doc.pdf")

    assert out is not None
    dumped = str(out)
    assert "HUGE PAGE TEXT" not in dumped and "more text" not in dumped
    assert out["structure"][0]["title"] == "1 Intro"
    assert out["structure"][0]["nodes"][0]["title"] == "1.1 Sub"


def test_document_structure_summary_only_on_outermost_level(tmp_path: Path):
    # Top section and its child both carry a summary; on an auto-expanded view the
    # outermost level keeps its summary but the deeper preview level drops it.
    structure = [
        {
            "title": "1 Intro",
            "node_id": "0001",
            "start_index": 1,
            "end_index": 4,
            "summary": "TOP SUMMARY",
            "nodes": [
                {
                    "title": "1.1 Sub",
                    "node_id": "0002",
                    "start_index": 1,
                    "end_index": 2,
                    "summary": "CHILD SUMMARY",
                }
            ],
        }
    ]
    write_doc(tmp_path, "doc", category=["a"], summary="x", structure=structure)

    out = store.document_structure(tmp_path, "doc.pdf")  # auto -> expands to level 2

    assert isinstance(out, dict)
    top = out["structure"][0]
    assert top["summary"] == "TOP SUMMARY"
    assert "summary" not in top["nodes"][0]  # deeper preview level trimmed


def _leaf(title: str, node_id: str, start: int, end: int) -> dict:
    return {"title": title, "node_id": node_id, "start_index": start, "end_index": end}


def _deep_doc(tmp_path: Path) -> str:
    # 2 top sections; section 1 has 2 subsections, one of which has a sub-subsection.
    structure = [
        {
            "title": "1 Methods",
            "node_id": "0001",
            "start_index": 1,
            "end_index": 10,
            "nodes": [
                {
                    "title": "1.1 Setup",
                    "node_id": "0002",
                    "start_index": 1,
                    "end_index": 4,
                    "nodes": [_leaf("1.1.1 Hardware", "0003", 2, 3)],
                },
                _leaf("1.2 Training", "0004", 5, 10),
            ],
        },
        _leaf("2 Results", "0005", 11, 20),
    ]
    write_doc(tmp_path, "deep", category=["a"], summary="x", structure=structure)
    return "deep.pdf"


def test_document_structure_explicit_max_depth_truncates(tmp_path: Path):
    doc = _deep_doc(tmp_path)
    out = store.document_structure(tmp_path, doc, max_depth=1)
    assert isinstance(out, dict)
    top = out["structure"]
    assert [n["title"] for n in top] == ["1 Methods", "2 Results"]
    # children hidden, but signalled
    assert "nodes" not in top[0]
    assert top[0]["subsection_count"] == 2
    assert "subsection_count" not in top[1]  # leaf has none


def test_document_structure_auto_depth_expands_shallow_tree(tmp_path: Path):
    doc = _deep_doc(tmp_path)
    out = store.document_structure(tmp_path, doc)  # auto
    assert isinstance(out, dict)
    methods = out["structure"][0]
    # small tree -> expands past level 1; level-2 children present
    assert {n["title"] for n in methods["nodes"]} == {"1.1 Setup", "1.2 Training"}


def test_document_structure_auto_depth_stays_shallow_when_wide(tmp_path: Path):
    # 60 top-level sections, each with one child: level-2 would exceed the budget.
    structure = [
        {
            "title": f"Sec {i}",
            "node_id": str(i).zfill(4),
            "start_index": i,
            "end_index": i,
            "nodes": [_leaf(f"Sub {i}", f"9{i:03d}", i, i)],
        }
        for i in range(1, 61)
    ]
    write_doc(tmp_path, "wide", category=["a"], summary="x", structure=structure)

    out = store.document_structure(tmp_path, "wide.pdf")

    assert isinstance(out, dict)
    assert all("nodes" not in n for n in out["structure"])
    assert all(n["subsection_count"] == 1 for n in out["structure"])


def test_document_structure_zoom_by_node_id(tmp_path: Path):
    doc = _deep_doc(tmp_path)
    out = store.document_structure(tmp_path, doc, node_id="0002")
    assert isinstance(out, dict)
    assert out["node_id"] == "0002"
    assert out["title"] == "1.1 Setup"
    assert [c["title"] for c in out["nodes"]] == ["1.1.1 Hardware"]


def test_document_structure_zoom_unknown_node(tmp_path: Path):
    doc = _deep_doc(tmp_path)
    out = store.document_structure(tmp_path, doc, node_id="9999")
    assert isinstance(out, str)
    assert "9999" in out


def test_document_structure_unknown_doc(corpus_dir: Path):
    assert store.document_structure(corpus_dir, "nope.pdf") is None


def test_resolve_pdf(corpus_dir: Path):
    assert store.resolve_pdf(corpus_dir, "fin_q3_2023.pdf") is not None
    assert store.resolve_pdf(corpus_dir, "missing.pdf") is None
