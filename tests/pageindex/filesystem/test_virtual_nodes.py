from pathlib import Path

from vlm_rag_index.pageindex.filesystem import store, virtual_nodes


def test_available_axes_requires_two_distinct_values(corpus_dir: Path):
    corpus = store.load_corpus(corpus_dir)
    axes = virtual_nodes.available_axes(corpus, ["category", "entities", "keywords"])
    assert "category" in axes  # 3 distinct categories
    assert "entities" in axes  # many distinct entities
    assert "keywords" not in axes  # all empty -> not usable


def test_project_bins_by_axis(corpus_dir: Path):
    corpus = store.load_corpus(corpus_dir)
    tree = virtual_nodes.project(corpus, "category")
    assert set(tree) == {"financial reporting", "machine learning", "legal"}
    assert len(tree["financial reporting"]) == 8
    assert "fin_q3_2023.pdf" in tree["financial reporting"]


def test_project_multi_valued_axis_appears_under_each():
    from vlm_rag_index.pageindex.filesystem.store import CorpusDoc
    from vlm_rag_index.pageindex.structured_responses import MetadataSidecar

    doc = CorpusDoc(
        doc_name="d.pdf",
        pdf=Path("d.pdf"),
        index_path=Path("d.json"),
        meta_path=Path("d.meta.json"),
        metadata=MetadataSidecar(
            doc_name="d.pdf",
            category=["a", "b"],
            entities=[],
            summary="",
            keywords=[],
        ),
    )
    tree = virtual_nodes.project([doc], "category")
    assert tree == {"a": ["d.pdf"], "b": ["d.pdf"]}


def test_folder_children_root_and_leaf(corpus_dir: Path):
    corpus = store.load_corpus(corpus_dir)
    root = virtual_nodes.folder_children(corpus, "/", ["category", "entities"])
    assert all(c["type"] == "folder" for c in root)
    assert {c["name"] for c in root} == {"financial reporting", "machine learning", "legal"}

    leaf = virtual_nodes.folder_children(corpus, "/legal", ["category", "entities"])
    assert all(c["type"] == "document" for c in leaf)
    assert len(leaf) == 6


def test_folder_summary_lists_documents(corpus_dir: Path):
    corpus = store.load_corpus(corpus_dir)
    summary = virtual_nodes.folder_summary(corpus, "/machine learning", ["category"])
    assert "machine learning" in summary
    assert "7 documents" in summary
    assert "ml_rlhf.pdf" in summary
