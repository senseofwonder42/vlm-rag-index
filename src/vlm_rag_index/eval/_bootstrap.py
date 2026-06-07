"""Redirect the corpus location before the agent layer is imported.

`Settings` is built once at import time (`core/config.py`), and the agent tools
read `settings.index_results_dir`. Pointing the agent at the dataset therefore
means setting `INDEX_RESULTS_DIR` *before* importing anything that pulls in
`settings`. Entry points call `set_results_dir` first, then import the agent.
"""

from __future__ import annotations

import os
from pathlib import Path


def set_results_dir(path: Path) -> None:
    """Set the `INDEX_RESULTS_DIR` env var so the agent reads from `path`.

    Must be called before importing `vlm_rag_index.agent` (or any module that
    imports `core.config`), or the override is ignored.
    """
    os.environ["INDEX_RESULTS_DIR"] = str(path)
