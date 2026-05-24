from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent),
    undefined=StrictUndefined,
    autoescape=False,
    keep_trailing_newline=True,
)


def render(name: str, **ctx: Any) -> str:
    return _env.get_template(name).render(**ctx)
