from vlm_rag_index.pageindex.structured_responses import PageHeading, PageInfo
from vlm_rag_index.pageindex.vlm.reconcile import _norm, _reconcile_records


def test_norm_strips_numbering_and_punct():
    assert _norm("2.3 Methods.") == "methods"
    assert _norm("A.1   Appendix-One!") == "appendix one"
    assert _norm("Conclusion") == "conclusion"


def test_reconcile_no_overlap_is_passthrough():
    records = [
        PageInfo(page_index=1, headings=[PageHeading(text="A", visual_rank="large")],
                 ending_sections=[]),
        PageInfo(page_index=2, headings=[], ending_sections=[]),
    ]
    out = _reconcile_records(records)
    assert out == records


def test_reconcile_off_by_one_picks_smaller_page():
    # Same heading reported on pages 3 and 4 across two overlapping batches.
    records = [
        PageInfo(page_index=3, headings=[PageHeading(text="Methods", visual_rank="large")],
                 ending_sections=[]),
        PageInfo(page_index=4, headings=[PageHeading(text="Methods", visual_rank="large")],
                 ending_sections=[]),
        PageInfo(page_index=3, headings=[], ending_sections=[]),
        PageInfo(page_index=4, headings=[], ending_sections=[]),
    ]
    out = _reconcile_records(records)
    by_page = {r.page_index: r for r in out}
    # Heading should appear only on the smaller (3) page (tie-break rule).
    assert any(h.text == "Methods" for h in by_page[3].headings)
    assert not any(h.text == "Methods" for h in by_page[4].headings)


def test_reconcile_picks_longest_paraphrase():
    # Two variants that normalise to the same string (numbering prefix stripped) -> cluster.
    records = [
        PageInfo(page_index=5,
                 headings=[PageHeading(text="Introduction", visual_rank="large")],
                 ending_sections=[]),
        PageInfo(page_index=5,
                 headings=[PageHeading(text="1 Introduction", visual_rank="large")],
                 ending_sections=[]),
        # Force overlap so the fast-path doesn't skip reconciliation.
        PageInfo(page_index=6, headings=[], ending_sections=[]),
        PageInfo(page_index=6, headings=[], ending_sections=[]),
    ]
    out = _reconcile_records(records)
    titles = [h.text for r in out for h in r.headings]
    assert titles == ["1 Introduction"]


def test_reconcile_merges_ending_sections_dedup():
    records = [
        PageInfo(page_index=7, headings=[], ending_sections=["1 Introduction"]),
        PageInfo(page_index=7, headings=[], ending_sections=["Introduction"]),
    ]
    out = _reconcile_records(records)
    assert len(out[0].ending_sections) == 1
