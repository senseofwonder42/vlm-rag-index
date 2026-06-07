"""Ask the agent a single question (entry point: `ask`).

Builds the ReAct agent over the dataset index and prints both the final answer
and the trace of which document/pages the agent actually read — handy for
eyeballing retrieval before running the full eval.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console

from vlm_rag_index.eval._bootstrap import set_results_dir
from vlm_rag_index.eval.paths import INDEX_DIR as DEFAULT_RESULTS_DIR


async def _run(question: str) -> int:
    # Imported lazily: build_graph() pulls in `settings`, which must already see
    # the redirected INDEX_RESULTS_DIR set by main() below.
    from vlm_rag_index.agent.graph import build_graph
    from vlm_rag_index.eval.agent_runner import run_query

    console = Console()
    run = await run_query(build_graph(), question)

    if run.error:
        console.print(f"[red]error:[/red] {run.error}")
        return 1

    console.print("[bold]Pages read[/bold]")
    if run.calls:
        for doc, pages in run.calls:
            console.print(f"  {doc}: {pages}")
    else:
        console.print("  [yellow](agent read no pages)[/yellow]")
    console.print("\n[bold]Answer[/bold]")
    console.print(run.final_answer or "[yellow](empty)[/yellow]")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse args, point the agent at the dataset, and run one query."""
    parser = argparse.ArgumentParser(prog="ask", description=__doc__)
    parser.add_argument("question", help="the question to ask the agent")
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help=f"directory of indexed PDFs (default: {DEFAULT_RESULTS_DIR})",
    )
    args = parser.parse_args(argv)

    set_results_dir(Path(args.results_dir))
    return asyncio.run(_run(args.question))


if __name__ == "__main__":
    sys.exit(main())
