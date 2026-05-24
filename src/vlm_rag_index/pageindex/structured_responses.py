from typing import Literal

from pydantic import BaseModel

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
