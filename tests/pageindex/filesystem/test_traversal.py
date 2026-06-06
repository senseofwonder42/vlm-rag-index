"""Adaptive traversal over the 21-document fixture corpus.

The LLM decisions are scripted so the navigation logic is tested deterministically,
without a real model.
"""

from pathlib import Path

import pytest
from pydantic import BaseModel

from vlm_rag_index.core.config import Settings
from vlm_rag_index.pageindex.filesystem import store, traversal, virtual_nodes


class ScriptedClient:
    """Returns pre-canned structured responses in order; records the schemas asked."""

    def __init__(self, responses: list[BaseModel]) -> None:
        self._responses = list(responses)
        self.schemas: list[str] = []

    async def acomplete_structured(self, messages, schema, **kwargs):  # noqa: ARG002
        self.schemas.append(schema.__name__)
        return self._responses.pop(0)


def _settings(**overrides) -> Settings:
    base = dict(
        llm_api_key="sk-test",
        filesystem_axes_default=["category", "entities"],
        filesystem_max_tree_depth=3,
        filesystem_flatten_threshold=0.6,
        filesystem_max_llm_hops_per_query=5,
    )
    base.update(overrides)
    return Settings(**base)


def _axis(name: str) -> virtual_nodes.AxisChoice:
    return virtual_nodes.AxisChoice(axis=name)


def _decision(labels: list[str], confidence: float) -> traversal.RelevanceDecision:
    return traversal.RelevanceDecision(relevant_labels=labels, confidence=confidence)


async def test_search_narrows_to_single_document(corpus_dir: Path):
    corpus = store.load_corpus(corpus_dir)
    client = ScriptedClient([_axis("entities"), _decision(["Q3 2023"], 0.9)])

    result = await traversal.search_filesystem(
        "Which document discusses Q3 revenue?",
        corpus,
        client=client,
        settings=_settings(),
    )

    assert result == ["fin_q3_2023.pdf"]
    assert client.schemas == ["AxisChoice", "RelevanceDecision"]


async def test_low_confidence_flattens_then_descends_next_axis(corpus_dir: Path):
    corpus = store.load_corpus(corpus_dir)
    client = ScriptedClient(
        [
            _axis("category"),
            _decision([], 0.2),  # unclear -> flatten, keep all, try next axis
            _axis("entities"),
            _decision(["Q3 2023"], 0.95),
        ]
    )

    result = await traversal.search_filesystem(
        "Which document discusses Q3 revenue?",
        corpus,
        client=client,
        settings=_settings(),
    )

    assert result == ["fin_q3_2023.pdf"]


async def test_relevant_category_narrows_candidate_set(corpus_dir: Path):
    corpus = store.load_corpus(corpus_dir)
    # One hop: pick category, keep only "legal" (6 docs), then stop (depth budget 1).
    client = ScriptedClient([_axis("category"), _decision(["legal"], 0.9)])

    result = await traversal.search_filesystem(
        "Find the NDA",
        corpus,
        client=client,
        settings=_settings(filesystem_max_tree_depth=1),
    )

    assert len(result) == 6
    assert set(result) == {
        "legal_nda_acme.pdf",
        "legal_msa_beta.pdf",
        "legal_lease_hq.pdf",
        "legal_employment.pdf",
        "legal_privacy.pdf",
        "legal_terms.pdf",
    }


async def test_hop_budget_caps_llm_calls(corpus_dir: Path):
    corpus = store.load_corpus(corpus_dir)
    # Budget of 1: only the axis-selection call fits; no narrowing happens.
    client = ScriptedClient([_axis("category")])

    result = await traversal.search_filesystem(
        "anything",
        corpus,
        client=client,
        settings=_settings(filesystem_max_llm_hops_per_query=1),
    )

    assert len(result) == 21
    assert client.schemas == ["AxisChoice"]


async def test_select_axis_clamps_unknown_axis(corpus_dir: Path):
    corpus = store.load_corpus(corpus_dir)
    client = ScriptedClient([_axis("not_a_real_axis")])

    axis = await virtual_nodes.select_axis(
        client, "q", ["category", "entities"], corpus
    )

    assert axis == "category"  # falls back to first offered axis


@pytest.mark.parametrize("query", ["Which document discusses Q3 revenue?"])
async def test_empty_corpus_returns_nothing(query: str):
    client = ScriptedClient([])
    result = await traversal.search_filesystem(
        query, [], client=client, settings=_settings()
    )
    assert result == []
