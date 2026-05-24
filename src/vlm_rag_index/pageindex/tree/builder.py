from typing import Any


def _count_nodes(tree: list[dict[str, Any]]) -> int:
    n = 0
    for node in tree:
        n += 1 + _count_nodes(node["nodes"])
    return n


def write_node_ids(tree: list[dict[str, Any]]) -> None:
    """Stamp depth-first zero-padded `node_id` strings on every node in-place.

    Width is `max(4, len(str(total_nodes)))` so the example doc with two nodes gets
    `0001`, `0002` and a 12 000-node corpus gets a wider stamp without padding-creep.
    """
    total = _count_nodes(tree)
    width = max(4, len(str(total)))
    counter = {"n": 0}

    def walk(nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            counter["n"] += 1
            node["node_id"] = str(counter["n"]).zfill(width)
            walk(node["nodes"])

    walk(tree)


def embed_text(tree: list[dict[str, Any]], pages_text: dict[int, str]) -> None:
    """For each LEAF node, set `text` to "\n\n".join of pages in its range."""

    def walk(nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            if not node["nodes"]:
                start = node["start_index"]
                end = node["end_index"]
                pages = [pages_text.get(p, "") for p in range(start, end + 1)]
                node["text"] = "\n\n".join(p for p in pages if p)
            else:
                walk(node["nodes"])

    walk(tree)
