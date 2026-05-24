from vlm_rag_index.pageindex.structured_responses import PageHeading, PageInfo
from vlm_rag_index.pageindex.vlm.extract import (
    _assign_end_indices,
    _build_tree,
    _dense_rank_map,
    _depth_from_numbering,
    _sliding_windows,
)


def test_sliding_windows_no_overlap():
    assert _sliding_windows([1, 2, 3, 4, 5], size=2, overlap=0) == [[1, 2], [3, 4], [5]]


def test_sliding_windows_overlap_clamped():
    # overlap >= size -> clamped to size-1.
    out = _sliding_windows([1, 2, 3, 4], size=2, overlap=5)
    # overlap clamps to 1 -> step 1.
    assert out == [[1, 2], [2, 3], [3, 4]]


def test_sliding_windows_empty():
    assert _sliding_windows([], size=4, overlap=0) == []


def test_sliding_windows_covers_full_sequence():
    seq = list(range(1, 11))
    windows = _sliding_windows(seq, size=3, overlap=1)
    covered = sorted({p for w in windows for p in w})
    assert covered == seq


def test_depth_from_numbering_simple():
    assert _depth_from_numbering("1 Introduction") == 1
    assert _depth_from_numbering("2.3 Methods") == 2
    assert _depth_from_numbering("4.1.7 Subsection") == 3
    assert _depth_from_numbering("A.1 Appendix") == 2


def test_depth_from_numbering_none():
    assert _depth_from_numbering("Conclusion") is None
    assert _depth_from_numbering("Abstract") is None


def test_dense_rank_map_skips_unseen():
    records = [
        PageInfo(
            page_index=1,
            headings=[PageHeading(text="Big", visual_rank="xlarge")],
            ending_sections=[],
        ),
        PageInfo(
            page_index=2,
            headings=[PageHeading(text="Tiny", visual_rank="small")],
            ending_sections=[],
        ),
    ]
    m = _dense_rank_map(records)
    # xlarge -> 1, small -> 2 (medium not seen, so skipped).
    assert m == {"xlarge": 1, "small": 2}


def _node(title, start, end=None, depth=1, nodes=None):
    return {
        "title": title,
        "start_index": start,
        "end_index": end,
        "nodes": nodes or [],
        "_depth": depth,
    }


def test_assign_end_indices_uses_vlm_signal():
    tree = [_node("Intro", 1), _node("Methods", 5), _node("Results", 9)]
    records = [
        PageInfo(page_index=4, headings=[], ending_sections=["Intro"]),
        PageInfo(page_index=8, headings=[], ending_sections=["Methods"]),
    ]
    _assign_end_indices(tree, records, last_page=15)
    assert tree[0]["end_index"] == 4
    assert tree[1]["end_index"] == 8
    # No VLM signal for Results -> sibling fallback to parent_end.
    assert tree[2]["end_index"] == 15


def test_assign_end_indices_sibling_handoff():
    tree = [_node("A", 1), _node("B", 5), _node("C", 9)]
    _assign_end_indices(tree, records=[], last_page=12)
    assert tree[0]["end_index"] == 5
    assert tree[1]["end_index"] == 9
    assert tree[2]["end_index"] == 12


def test_assign_end_indices_clamps_to_start():
    # A bogus ending signal at a page BEFORE the section's start must clamp.
    tree = [_node("Methods", 10)]
    records = [
        PageInfo(page_index=3, headings=[], ending_sections=["Methods"]),
    ]
    _assign_end_indices(tree, records, last_page=20)
    # The bogus signal at page 3 < start (10) is rejected (page < start), so the
    # sibling fallback kicks in.
    assert tree[0]["end_index"] == 20


def test_build_tree_synthesises_front_matter():
    records = [
        PageInfo(
            page_index=1,
            headings=[],
            ending_sections=[],
        ),
        PageInfo(
            page_index=3,
            headings=[PageHeading(text="1 Introduction", visual_rank="large")],
            ending_sections=[],
        ),
    ]
    tree = _build_tree(records, _dense_rank_map(records), last_page=10)
    assert tree[0]["title"] == "Front Matter"
    assert tree[0]["start_index"] == 1
    assert tree[0]["end_index"] == 2
    assert tree[1]["title"] == "1 Introduction"
    assert tree[1]["start_index"] == 3


def test_build_tree_no_heading_fallback():
    records = [
        PageInfo(page_index=1, headings=[], ending_sections=[]),
        PageInfo(page_index=2, headings=[], ending_sections=[]),
    ]
    tree = _build_tree(records, {}, last_page=2)
    assert len(tree) == 1
    assert tree[0]["title"] == "Document"
    assert tree[0]["start_index"] == 1
    assert tree[0]["end_index"] == 2


def test_build_tree_drops_caption_headings():
    records = [
        PageInfo(
            page_index=1,
            headings=[
                PageHeading(text="Figure 1: Architecture", visual_rank="medium"),
                PageHeading(text="1 Introduction", visual_rank="large"),
            ],
            ending_sections=[],
        ),
    ]
    tree = _build_tree(records, _dense_rank_map(records), last_page=2)
    titles = [n["title"] for n in tree]
    assert "1 Introduction" in titles
    assert not any("Figure" in t for t in titles)


def test_build_tree_depth_nesting():
    records = [
        PageInfo(
            page_index=1,
            headings=[PageHeading(text="1 Intro", visual_rank="large")],
            ending_sections=[],
        ),
        PageInfo(
            page_index=2,
            headings=[PageHeading(text="1.1 Setup", visual_rank="medium")],
            ending_sections=[],
        ),
        PageInfo(
            page_index=4,
            headings=[PageHeading(text="2 Results", visual_rank="large")],
            ending_sections=[],
        ),
    ]
    tree = _build_tree(records, _dense_rank_map(records), last_page=10)
    assert len(tree) == 2
    assert tree[0]["title"] == "1 Intro"
    assert len(tree[0]["nodes"]) == 1
    assert tree[0]["nodes"][0]["title"] == "1.1 Setup"
    assert tree[1]["title"] == "2 Results"
