from typing import Any, Literal, Protocol, TypedDict, TypeVar, runtime_checkable

from pydantic import BaseModel


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class ImageUrl(TypedDict):
    url: str


class ImageUrlPart(TypedDict):
    type: Literal["image_url"]
    image_url: ImageUrl


ContentPart = TextPart | ImageUrlPart


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]


T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, messages: list[Message], **kwargs: Any) -> str: ...

    async def acomplete(self, messages: list[Message], **kwargs: Any) -> str: ...

    def complete_structured(self, messages: list[Message], schema: type[T], **kwargs: Any) -> T: ...

    async def acomplete_structured(
        self, messages: list[Message], schema: type[T], **kwargs: Any
    ) -> T: ...

    def count_tokens(self, text: str) -> int: ...
