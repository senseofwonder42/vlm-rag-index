from vlm_rag_index.eval.manifest import ManifestRow


def _row(**overrides):
    base = dict(
        pdf="pdfs/hr_handbook_p75-79.pdf",
        query="q",
        query_id="q1",
        answer="a",
        answer_page_index=0,
        domain="hr",
        language="english",
        relevance_score=2,
    )
    base.update(overrides)
    return ManifestRow(**base)


def test_gold_doc_strips_pdfs_prefix():
    assert _row().gold_doc == "hr_handbook_p75-79.pdf"


def test_gold_pages_int_is_converted_to_1based():
    assert _row(answer_page_index=0).gold_pages_1based == [1]
    assert _row(answer_page_index=3).gold_pages_1based == [4]


def test_gold_pages_list_is_converted_to_1based():
    assert _row(answer_page_index=[0, 2]).gold_pages_1based == [1, 3]


def test_extra_manifest_fields_are_ignored():
    row = ManifestRow.model_validate(
        {
            "pdf": "pdfs/x.pdf",
            "query": "q",
            "query_id": "q2",
            "answer": "a",
            "answer_page_index": 1,
            "domain": "finance_en",
            "language": "english",
            "relevance_score": 2,
            "unexpected": "field",
        }
    )
    assert row.gold_pages_1based == [2]
