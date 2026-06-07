from vlm_rag_index.eval.inject import link_pdfs
from vlm_rag_index.eval.paths import index_dir_for


def test_index_dir_is_sibling_of_pdfs(tmp_path):
    pdfs = tmp_path / "sample" / "pdfs"
    assert index_dir_for(pdfs) == tmp_path / "sample" / "index"


def test_link_pdfs_creates_symlinks_to_sources(tmp_path):
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    (pdfs / "a.pdf").write_bytes(b"%PDF-1")
    (pdfs / "b.pdf").write_bytes(b"%PDF-2")
    index = tmp_path / "index"

    count = link_pdfs(pdfs, index)

    assert count == 2
    link = index / "a.pdf"
    assert link.is_symlink()
    assert link.resolve() == (pdfs / "a.pdf").resolve()
    assert link.read_bytes() == b"%PDF-1"


def test_link_pdfs_is_idempotent(tmp_path):
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    (pdfs / "a.pdf").write_bytes(b"%PDF")
    index = tmp_path / "index"

    assert link_pdfs(pdfs, index) == 1
    assert link_pdfs(pdfs, index) == 1  # second run does not raise
    assert (index / "a.pdf").is_symlink()
