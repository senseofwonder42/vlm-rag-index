"""Fixtures: materialise `{pdf, json, meta.json}` triples into a temp directory."""

import json
from pathlib import Path

import pytest


def write_doc(
    results_dir: Path,
    stem: str,
    *,
    category: list[str],
    entities: list[str] = (),
    keywords: list[str] = (),
    summary: str = "",
    structure: list[dict] | None = None,
    pdf: bool = True,
    meta: bool = True,
) -> str:
    """Write one document's files into `results_dir`. Returns its doc_name."""
    doc_name = f"{stem}.pdf"
    if pdf:
        (results_dir / doc_name).write_bytes(b"%PDF-1.4 fake\n")
    index = {
        "doc_name": doc_name,
        "structure": structure
        or [{"title": "1 Intro", "start_index": 1, "end_index": 2}],
    }
    (results_dir / f"{stem}.json").write_text(json.dumps(index), encoding="utf-8")
    if meta:
        sidecar = {
            "doc_name": doc_name,
            "category": list(category),
            "entities": list(entities),
            "summary": summary,
            "keywords": list(keywords),
            "schema_version": 1,
        }
        (results_dir / f"{stem}.meta.json").write_text(
            json.dumps(sidecar), encoding="utf-8"
        )
    return doc_name


# A corpus of 21 documents across three obvious categories, with a distinctive
# Q3-revenue document for the corpus-wide retrieval scenario.
_SPECS = [
    # financial reporting (8)
    ("fin_annual_2021", ["financial reporting"], ["Acme Corp", "FY2021"], "Acme 2021 annual report."),
    ("fin_annual_2022", ["financial reporting"], ["Acme Corp", "FY2022"], "Acme 2022 annual report."),
    ("fin_annual_2023", ["financial reporting"], ["Acme Corp", "FY2023"], "Acme 2023 annual report."),
    ("fin_q1_2023", ["financial reporting"], ["Acme Corp", "Q1 2023"], "Acme Q1 2023 results."),
    ("fin_q2_2023", ["financial reporting"], ["Acme Corp", "Q2 2023"], "Acme Q2 2023 results."),
    ("fin_q3_2023", ["financial reporting"], ["Acme Corp", "Q3 2023"], "Acme Q3 2023 revenue."),
    ("fin_q4_2023", ["financial reporting"], ["Acme Corp", "Q4 2023"], "Acme Q4 2023 results."),
    ("fin_budget_2024", ["financial reporting"], ["Acme Corp", "FY2024"], "Acme 2024 budget plan."),
    # machine learning (7)
    ("ml_transformers", ["machine learning"], ["Transformer"], "Survey of transformer architectures."),
    ("ml_diffusion", ["machine learning"], ["Diffusion"], "Diffusion models for image synthesis."),
    ("ml_rlhf", ["machine learning"], ["RLHF"], "Reinforcement learning from human feedback."),
    ("ml_embeddings", ["machine learning"], ["Embeddings"], "Text embedding methods compared."),
    ("ml_pruning", ["machine learning"], ["Pruning"], "Network pruning techniques."),
    ("ml_quantization", ["machine learning"], ["Quantization"], "Post-training quantization study."),
    ("ml_retrieval", ["machine learning"], ["RAG"], "Retrieval-augmented generation overview."),
    # legal (6)
    ("legal_nda_acme", ["legal"], ["Acme Corp"], "Non-disclosure agreement with Acme Corp."),
    ("legal_msa_beta", ["legal"], ["Beta LLC"], "Master services agreement with Beta LLC."),
    ("legal_lease_hq", ["legal"], ["Acme Corp"], "Headquarters office lease."),
    ("legal_employment", ["legal"], ["Acme Corp"], "Standard employment agreement template."),
    ("legal_privacy", ["legal"], ["Acme Corp"], "Privacy policy and data handling terms."),
    ("legal_terms", ["legal"], ["Acme Corp"], "Website terms of service."),
]


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """A directory holding 21 complete document triples across 3 categories."""
    for stem, category, entities, summary in _SPECS:
        write_doc(
            tmp_path,
            stem,
            category=category,
            entities=entities,
            keywords=[],
            summary=summary,
        )
    return tmp_path
