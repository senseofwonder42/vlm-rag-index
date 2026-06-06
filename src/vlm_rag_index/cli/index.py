import argparse
import asyncio
import json
import sys
from pathlib import Path

from loguru import logger
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from vlm_rag_index.core.config import settings
from vlm_rag_index.core.logging import setup_logging
from vlm_rag_index.pageindex.pipeline import agenerate_metadata, apage_index, to_dict


def _output_path(pdf: Path, output_dir: Path | None) -> Path:
    target_dir = output_dir if output_dir is not None else pdf.parent
    return target_dir / f"{pdf.stem}.json"


async def _index_one(pdf: Path, output_dir: Path | None, write_metadata: bool) -> Path:
    result = await apage_index(pdf)
    out_path = _output_path(pdf, output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(to_dict(result), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if write_metadata:
        sidecar = await agenerate_metadata(result)
        meta_path = out_path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(sidecar.model_dump(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("wrote {}", meta_path)
    return out_path


def _collect_pdfs(args: argparse.Namespace) -> list[Path]:
    if args.input_dir is not None:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            raise SystemExit(f"--input-dir not found: {input_dir}")
        return sorted(input_dir.glob("*.pdf"))
    if not args.pdf:
        raise SystemExit("usage: index <pdf> | --input-dir <dir>")
    pdf = Path(args.pdf)
    if not pdf.is_file():
        raise SystemExit(f"pdf not found: {pdf}")
    return [pdf]


async def _run(pdfs: list[Path], output_dir: Path | None, write_metadata: bool) -> int:
    if not pdfs:
        logger.warning("no PDFs to index")
        return 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("indexing", total=len(pdfs))
        for pdf in pdfs:
            progress.update(task, description=f"indexing {pdf.name}")
            out = await _index_one(pdf, output_dir, write_metadata)
            logger.info("wrote {}", out)
            progress.advance(task)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="index", description="VLM-index one or more PDFs.")
    parser.add_argument("pdf", nargs="?", help="path to a PDF (omit when using --input-dir)")
    parser.add_argument("--input-dir", help="batch-index every *.pdf in this directory")
    parser.add_argument(
        "--output-dir",
        help="write JSONs to this directory (default: next to each source PDF)",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="skip writing the .meta.json sidecar next to each index JSON",
    )
    args = parser.parse_args(argv)

    setup_logging(settings.log_level)
    pdfs = _collect_pdfs(args)
    output_dir = Path(args.output_dir) if args.output_dir else None
    write_metadata = settings.index_add_metadata and not args.no_metadata
    return asyncio.run(_run(pdfs, output_dir, write_metadata))


if __name__ == "__main__":
    sys.exit(main())
