import pytest

from vlm_rag_index.pageindex.llm.retry import exponential_backoff


def test_grows_with_attempt():
    a0 = exponential_backoff(0, base=1.0)
    a3 = exponential_backoff(3, base=1.0)
    assert a0 >= 1.0
    assert a3 >= 8.0
    assert a3 > a0


def test_respects_cap():
    big = exponential_backoff(20, base=1.0, cap=5.0)
    assert big <= 5.0 * 1.1 + 0.001


def test_negative_attempt_raises():
    with pytest.raises(ValueError):
        exponential_backoff(-1)
