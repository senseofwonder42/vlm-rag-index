from vlm_rag_index.pageindex.tree.post_process import reorder_keys, strip_internal_fields


def test_strip_internal_fields_recurses():
    tree = [
        {"title": "A", "start_index": 1, "end_index": 5, "_depth": 1, "nodes": [
            {"title": "A.1", "start_index": 1, "end_index": 3, "_depth": 2, "nodes": []},
        ]}
    ]
    strip_internal_fields(tree)
    assert "_depth" not in tree[0]
    assert "_depth" not in tree[0]["nodes"][0]


def test_reorder_keys_canonical_order_and_drops_none():
    node = {
        "nodes": [],
        "summary": None,
        "title": "T",
        "node_id": "0001",
        "end_index": 5,
        "start_index": 1,
        "text": "body",
    }
    out = reorder_keys(node)
    assert list(out.keys()) == ["title", "node_id", "start_index", "end_index", "text", "nodes"]
    assert "summary" not in out  # None drops


def test_reorder_keys_recurses_into_children():
    node = {"title": "P", "start_index": 1, "end_index": 5, "nodes": [
        {"nodes": [], "title": "C", "start_index": 1, "end_index": 5},
    ]}
    out = reorder_keys(node)
    child = out["nodes"][0]
    assert list(child.keys()) == ["title", "start_index", "end_index", "nodes"]
