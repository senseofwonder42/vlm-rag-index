from vlm_rag_index.eval.agent_runner import AgentRun
from vlm_rag_index.eval.manifest import ManifestRow
from vlm_rag_index.eval.metrics import score_row, summarize


def _row(query_id="q1", domain="hr", language="english", page=0):
    return ManifestRow(
        pdf="pdfs/doc.pdf",
        query="q",
        query_id=query_id,
        answer="a",
        answer_page_index=page,
        domain=domain,
        language=language,
        relevance_score=2,
    )


def test_right_doc_and_page_hits_both():
    # page=0 (0-based) -> gold page 1 (1-based), which the agent read.
    scored = score_row(_row(page=0), AgentRun(calls=[("doc.pdf", [1, 2])]))
    assert scored["doc_hit"] and scored["page_hit"]


def test_right_doc_wrong_page_is_doc_hit_only():
    scored = score_row(_row(page=0), AgentRun(calls=[("doc.pdf", [3, 4])]))
    assert scored["doc_hit"]
    assert not scored["page_hit"]


def test_wrong_doc_hits_neither():
    scored = score_row(_row(page=0), AgentRun(calls=[("other.pdf", [1])]))
    assert not scored["doc_hit"]
    assert not scored["page_hit"]


def test_no_calls_hits_neither():
    scored = score_row(_row(), AgentRun(calls=[]))
    assert not scored["doc_hit"]
    assert not scored["page_hit"]


def test_summarize_rates_and_breakdown():
    scored = [
        score_row(_row("a", domain="hr"), AgentRun(calls=[("doc.pdf", [1])])),  # both
        score_row(_row("b", domain="hr"), AgentRun(calls=[("doc.pdf", [9])])),  # doc only
        score_row(_row("c", domain="finance_en"), AgentRun(calls=[("x.pdf", [1])])),  # neither
    ]
    summary = summarize(scored)

    assert summary["n"] == 3
    assert summary["doc_hit_rate"] == 2 / 3
    assert summary["page_hit_rate"] == 1 / 3
    assert summary["page_hit_rate_given_doc_hit"] == 1 / 2
    assert summary["by_domain"]["hr"]["n"] == 2
    assert summary["by_domain"]["hr"]["page_hit_rate"] == 1 / 2
