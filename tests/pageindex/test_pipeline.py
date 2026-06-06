from vlm_rag_index.core.constants import METADATA_SCHEMA_VERSION
from vlm_rag_index.pageindex.pipeline import agenerate_metadata
from vlm_rag_index.pageindex.structured_responses import DocMetadata, MetadataSidecar
from vlm_rag_index.pageindex.tree.types import IndexResult, TreeNode


class _MetadataClient:
    """Records the prompt it was asked with and returns a canned DocMetadata."""

    def __init__(self, metadata: DocMetadata) -> None:
        self._metadata = metadata
        self.last_messages: list = []
        self.last_schema: type | None = None

    def complete(self, messages, **kwargs):  # noqa: ARG002
        return ""

    async def acomplete(self, messages, **kwargs):  # noqa: ARG002
        return ""

    def complete_structured(self, messages, schema, **kwargs):  # noqa: ARG002
        return self._metadata

    async def acomplete_structured(self, messages, schema, **kwargs):  # noqa: ARG002
        self.last_messages = messages
        self.last_schema = schema
        return self._metadata

    def count_tokens(self, text):  # noqa: ARG002
        return 0


def _result() -> IndexResult:
    return IndexResult(
        doc_name="annual_report_2023.pdf",
        structure=[
            TreeNode(title="1 Operations", start_index=1, end_index=4, summary="Ops review."),
            TreeNode(title="2 Financials", start_index=5, end_index=9, summary="The numbers."),
        ],
    )


async def test_agenerate_metadata_stamps_doc_name_and_schema_version():
    canned = DocMetadata(
        category=["financial reporting"],
        entities=["Acme Corp"],
        summary="Acme Corp's 2023 annual report.",
        keywords=["revenue"],
    )
    client = _MetadataClient(canned)

    sidecar = await agenerate_metadata(_result(), client=client)

    assert sidecar.doc_name == "annual_report_2023.pdf"
    assert sidecar.schema_version == METADATA_SCHEMA_VERSION
    assert sidecar.category == ["financial reporting"]
    assert sidecar.entities == ["Acme Corp"]
    assert sidecar.keywords == ["revenue"]
    assert client.last_schema is DocMetadata


async def test_agenerate_metadata_prompts_with_top_level_sections():
    client = _MetadataClient(
        DocMetadata(category=[], entities=[], summary="", keywords=[]),
    )

    await agenerate_metadata(_result(), client=client)

    prompt = client.last_messages[0]["content"]
    assert "annual_report_2023.pdf" in prompt
    assert "1 Operations" in prompt
    assert "Ops review." in prompt
    assert "2 Financials" in prompt


def test_metadata_sidecar_key_order_matches_spec():
    sidecar = MetadataSidecar.from_metadata(
        "doc.pdf",
        DocMetadata(category=["a"], entities=["b"], summary="s", keywords=["k"]),
    )
    assert list(sidecar.model_dump().keys()) == [
        "doc_name",
        "category",
        "entities",
        "summary",
        "keywords",
        "schema_version",
    ]
