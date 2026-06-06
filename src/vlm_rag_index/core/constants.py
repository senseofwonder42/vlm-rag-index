from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

# Bumped when the `.meta.json` sidecar schema changes so v2 can migrate lazily.
METADATA_SCHEMA_VERSION: int = 1
