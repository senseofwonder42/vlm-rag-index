from vlm_rag_index.pageindex.tree.builder import embed_text, write_node_ids


def test_write_node_ids_depth_first():
    tree = [
        {"title": "A", "start_index": 1, "end_index": 5, "nodes": [
            {"title": "A.1", "start_index": 1, "end_index": 3, "nodes": []},
            {"title": "A.2", "start_index": 4, "end_index": 5, "nodes": []},
        ]},
        {"title": "B", "start_index": 6, "end_index": 10, "nodes": []},
    ]
    write_node_ids(tree)
    assert tree[0]["node_id"] == "0001"
    assert tree[0]["nodes"][0]["node_id"] == "0002"
    assert tree[0]["nodes"][1]["node_id"] == "0003"
    assert tree[1]["node_id"] == "0004"


def test_embed_text_only_on_leaves():
    pages = {1: "page one", 2: "page two", 3: "page three"}
    tree = [
        {"title": "Parent", "start_index": 1, "end_index": 3, "nodes": [
            {"title": "Leaf1", "start_index": 1, "end_index": 2, "nodes": []},
            {"title": "Leaf2", "start_index": 3, "end_index": 3, "nodes": []},
        ]},
    ]
    embed_text(tree, pages)
    assert tree[0].get("text") is None  # parent untouched
    assert tree[0]["nodes"][0]["text"] == "page one\n\npage two"
    assert tree[0]["nodes"][1]["text"] == "page three"
