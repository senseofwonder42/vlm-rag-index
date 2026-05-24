import base64

from vlm_rag_index.pageindex.pdf.renderer import image_part


def test_image_part_shape():
    png = b"\x89PNG\r\n\x1a\nfake"
    part = image_part(png)
    assert part["type"] == "image_url"
    url = part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    decoded = base64.b64decode(url.split(",", 1)[1])
    assert decoded == png
