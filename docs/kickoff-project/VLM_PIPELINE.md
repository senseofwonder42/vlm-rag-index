# VLM indexing pipeline

How a PDF becomes an `IndexResult` JSON.

## Inputs and output

**Input**: a PDF (path or `BytesIO`), an `LLMClient` instance, the `Settings` singleton.

**Output**: an `IndexResult` JSON file written beside the PDF:

```json
{
  "doc_name": "example.pdf",
  "structure": [
    {
      "title": "Chapter 1",
      "node_id": "0001",
      "start_index": 1,
      "end_index": 50,
      "summary": "...",
      "text": "...",
      "nodes": [
        {
          "title": "Section 1.1",
          "node_id": "0002",
          "start_index": 1,
          "end_index": 12,
          "nodes": []
        }
      ]
    }
  ],
  "doc_description": "..."
}
```

`node_id`, `summary`, `text`, and `doc_description` appear only when the corresponding `index_add_*` setting is true. Page indices are **1-based**.

## Stage 1 — Render

`pdf.renderer.render_pages(source, indices, dpi=settings.vlm_dpi)` opens the PDF with PyMuPDF and returns `{page_idx: PNG bytes}`. The long edge is capped at 1568 pixels — beyond that, providers downscale internally and we pay upload bandwidth for nothing. 144 DPI hits this cap on A4 and is the default. Raise to 192 for dense academic papers where small subsection headings are being missed.

PNG bytes are wrapped as OpenAI-compatible `image_url` parts via `image_part(png_bytes)`:

```python
{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
```

## Stage 2 — Slide and call

The page-index list is split into windows by `_sliding_windows(seq, size, overlap)`:

- `size = min(vlm_pages_per_batch, vlm_max_images_per_call)`. Provider caps (e.g., OpenRouter→Nvidia: 10 images/request) win.
- `overlap = vlm_window_overlap`, clamped to `[0, size - 1]`.

`overlap = 0` (default) is cheapest: non-overlapping batches, one call per `size` pages. `overlap = size // 2` enables sliding-window batching that corrects intra-batch "heading reported one page late" errors via reconciliation, at proportional extra cost.

For each window, `vlm.extract._call_batch` sends `llm.acomplete_structured(messages, PageBatchResponse)` where `messages` is one user message containing the rendered prompt text followed by one `image_part` per page in the window, in page-index order.

Records whose `page_index` is outside the requested batch are filtered defensively before reconciliation; a warning is logged if any are dropped.

## Stage 3 — Prompt

`prompts/vlm_page_batch.j2`. Variables: `num_pages` (int), `page_indices` (list[int]).

The prompt:

- Declares the images are contiguous pages in the order shown.
- Asks for one `PageInfo` record per image.
- Defines the four fields (`page_index`, `headings`, `ending_sections`, `description`) and the `visual_rank` scale (`xsmall` through `xlarge`, with explicit guidance that the same visual size class must map to the same rank consistently across pages).
- **Explicitly excludes** figure/table/algorithm/equation/listing/scheme captions, page numbers, running headers/footers, author names, affiliations, abstract labels, and bold inline emphasis. These are the consistent false-positive heading classes; enumerating them in the prompt eliminates ~90% of them.
- Forbids putting a heading in both `headings` and `ending_sections` for the same page — `ending_sections` is for sections that began on a *previous* page.

Output is one `PageBatchResponse`: `{"pages": [PageInfo, ...]}`.

## Stage 4 — Reconcile (only if `vlm_window_overlap > 0`)

`vlm.reconcile._reconcile_records(records)` collapses overlapping windows into one canonical `PageInfo` per page:

1. **Cluster headings across all records**. Two heading texts are in the same cluster if `SequenceMatcher(norm(a), norm(b)).ratio() >= 0.92`. `norm()` lowercases, strips section-number prefixes (`2.3 Methods` → `methods`), removes punctuation, and collapses whitespace.
2. **Pick the canonical page for each cluster** by majority vote across the cluster's source records. Ties go to the smaller page number — the model's bias is toward reporting too late, never too early.
3. **Pick the canonical heading text** as the longest variant seen; model paraphrases tend to truncate.
4. **Pick the canonical visual rank** as the cluster mode.
5. **Merge per-page descriptions** by taking the longest non-empty one; **merge `ending_sections`** as a deduplicated union (using `norm()` for dedup).

When every page appears in exactly one record (the common `overlap = 0` case), reconciliation is a no-op early-exit.

## Stage 5 — Build the tree

`vlm.extract._build_tree(records, rank_to_depth, last_page)` walks records in page order and pushes/pops a parent stack to produce nested dicts:

- **Depth from numbering first**: a heading whose text starts with `1.`, `1.2`, `1.2.3`, or `A.1` gets depth equal to the dot-count + 1. This is the most reliable signal when present.
- **Visual-rank fallback**: when numbering is absent, depth comes from a dense ranking of the visual ranks actually observed in the document. `xlarge` (if seen) → depth 1, next-largest seen → depth 2, and so on. Ranks that never appear are skipped, so the depth values are contiguous integers — the tree builder expects no gaps.
- **Caption filter**: a regex on `^(table|figure|fig\.|algorithm|alg\.|equation|eq\.|listing|scheme)\s*\d` drops anything the prompt's exclusion list missed.
- **Synthetic Front Matter**: when the first detected heading is on page 2 or later, a `Front Matter` leaf node spans pages 1..(first_heading_page - 1). Title pages, abstracts, and the PDF's own TOC live there.
- **No-heading fallback**: if no headings are detected anywhere, the whole document becomes a single leaf titled `Document` spanning all pages.

## Stage 6 — Assign `end_index`

`_assign_end_indices()` runs in two coordinated passes:

1. **VLM signal pass** (`_build_section_end_map`): every `ending_sections` title across all records is fuzzy-matched to a node title in the current tree (`SequenceMatcher.ratio() >= 0.85`). When a node matches on multiple pages, take the latest — under-detection of section ends is the dominant failure mode, so be generous. Signal-derived ends are clamped to `>= start_index`.
2. **Sibling-handoff fallback**: a node without a VLM signal ends at the next sibling's `start_index`, or at its parent's `end_index` if it is the last sibling.

The VLM signal always wins when present and consistent. Front Matter's `end_index` is set explicitly during tree assembly and is not overridden here.

## Stage 7 — Post-process

In order:

- **Node IDs** (`index_add_node_id`, default `true`): zero-padded depth-first sequence stamped on every node. No LLM call.
- **Leaf summaries** (`index_add_node_summary`): one LLM call per leaf, prompted with the per-page descriptions already collected from the VLM batch step. No fresh page reads. Then **parent summaries**: bottom-up, each non-leaf reduces its children's summaries into one.
- **Embedded text** (`index_add_node_text`): a single PyMuPDF text pass extracts plain text per page; each leaf gets `text = "\n\n".join(pages_in_range)`. Significantly enlarges the JSON. Leave off unless the agent's `answer_from_pages` is being bypassed.
- **Strip transient fields**: `_depth` and `_page_descriptions` are removed before serialization.
- **Reorder keys**: `title`, `node_id`, `start_index`, `end_index`, `summary`, `text`, `nodes` — deterministic order so diffs of regenerated JSONs stay reviewable.
- **Document description** (`index_add_doc_description`): one final LLM call that condenses root-level node summaries into a single sentence at `IndexResult.doc_description`. Requires summaries to have run first.

## Optional heading verification

When `vlm_verify_heading_text = true`, an extra PyMuPDF text pass extracts page text and replaces every VLM-emitted heading string with the closest text-layer line (`SequenceMatcher.ratio() >= 0.8`). Defends against the model paraphrasing a heading rather than quoting it. Costs one local text-extraction pass; no extra LLM call. Off by default — modern vision models hallucinate heading text rarely enough that the overhead isn't worth it for most documents.

## Structured response models

```python
VisualRank = Literal["xsmall", "small", "medium", "large", "xlarge"]

class PageHeading(BaseModel):
    text: str
    visual_rank: VisualRank

class PageInfo(BaseModel):
    page_index: int
    headings: list[PageHeading]
    ending_sections: list[str]
    description: str

class PageBatchResponse(BaseModel):
    pages: list[PageInfo]
```

These are the wire format between the prompt and the pipeline. Pydantic validates them on the way in; everything downstream uses them as immutable inputs.

## Knobs at a glance

| Setting | Default | Knob effect |
|---|---|---|
| `vlm_dpi` | `144` | Render resolution. Higher → better small-text legibility, more image tokens. |
| `vlm_pages_per_batch` | `8` | Pages per LLM call. Higher → fewer calls, more tokens per call. |
| `vlm_max_images_per_call` | `8` | Provider hard cap. Effective batch = `min(per_batch, max_per_call)`. |
| `vlm_window_overlap` | `0` | Pages shared between batches. `0` = cheapest; `~half-batch` = robust to off-by-one. |
| `vlm_verify_heading_text` | `false` | Snap VLM headings to text-layer matches. |
| `index_add_node_id` | `true` | Depth-first zero-padded IDs. |
| `index_add_node_summary` | `false` | Two-pass summarisation (leaf then parent). |
| `index_add_node_text` | `false` | Embed page text in leaves. |
| `index_add_doc_description` | `false` | One-sentence doc summary at root. Requires summaries. |
