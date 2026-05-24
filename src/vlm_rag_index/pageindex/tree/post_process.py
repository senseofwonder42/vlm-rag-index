from typing import Any

_INTERNAL_FIELDS = ("_depth",)
_KEY_ORDER = ("title", "node_id", "start_index", "end_index", "summary", "text", "nodes")


def strip_internal_fields(tree: list[dict[str, Any]]) -> None:
    """Remove transient `_depth` / `_page_descriptions` from every node in-place."""

    def walk(nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            for field in _INTERNAL_FIELDS:
                node.pop(field, None)
            walk(node["nodes"])

    walk(tree)


def reorder_keys(node: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `node` with keys in the canonical order; recurses into children."""
    out: dict[str, Any] = {}
    for key in _KEY_ORDER:
        if key not in node:
            continue
        value = node[key]
        if key == "nodes":
            out[key] = [reorder_keys(c) for c in value]
        elif value is None:
            continue  # drop unset optional fields for stable JSON diffs
        else:
            out[key] = value
    # Carry any keys not in _KEY_ORDER (defensive; should not happen post-strip).
    for key in node:
        if key in out or key in _INTERNAL_FIELDS or key in _KEY_ORDER:
            continue
        out[key] = node[key]
    return out
