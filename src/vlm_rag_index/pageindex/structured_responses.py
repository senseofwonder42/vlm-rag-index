from typing import Literal

from pydantic import BaseModel, Field

from vlm_rag_index.core.constants import METADATA_SCHEMA_VERSION

VisualRank = Literal["xsmall", "small", "medium", "large", "xlarge"]


class PageHeading(BaseModel):
    text: str
    visual_rank: VisualRank


class PageInfo(BaseModel):
    page_index: int
    headings: list[PageHeading]
    ending_sections: list[str]


class PageBatchResponse(BaseModel):
    pages: list[PageInfo]


class DocMetadata(BaseModel):
    """The LLM-generated fields of a per-document `.meta.json` sidecar.

    `doc_name` and `schema_version` are stamped by the pipeline, not the model —
    see `MetadataSidecar`.
    """

    category: list[str] = Field(
        description="Broad subject categories, most general first (e.g. "
        '["financial reporting", "annual"]).',
    )
    entities: list[str] = Field(
        description="Named organisations, products, people, or periods the document is about.",
    )
    summary: str = Field(
        description="One or two sentences describing what the document is about.",
    )
    keywords: list[str] = Field(
        description="Salient terms a reader might search for.",
    )


class MetadataSidecar(BaseModel):
    """The full `.meta.json` written next to each index JSON.

    Field declaration order is the on-disk key order; keep it in sync with the
    example in FILESYSTEM.md.
    """

    doc_name: str
    category: list[str]
    entities: list[str]
    summary: str
    keywords: list[str]
    schema_version: int = METADATA_SCHEMA_VERSION

    @classmethod
    def from_metadata(cls, doc_name: str, metadata: DocMetadata) -> "MetadataSidecar":
        return cls(doc_name=doc_name, **metadata.model_dump())
