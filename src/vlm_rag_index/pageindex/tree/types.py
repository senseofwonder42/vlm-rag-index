from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TreeNode(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str
    node_id: str | None = None
    start_index: int
    end_index: int
    summary: str | None = None
    text: str | None = None
    nodes: list[TreeNode] = []


class IndexResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    doc_name: str
    structure: list[TreeNode]
    doc_description: str | None = None
