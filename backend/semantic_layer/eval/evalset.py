"""Load the golden-question evalset."""

import json
from pathlib import Path

from semantic_layer.config import settings

_DEFAULT_PATH = "data/eval/golden_questions.json"


def load_evalset(path: str | None = None) -> list[dict]:
    """Load evalset items [{id, question, expect}, ...] from JSON.

    Default path is resolved relative to the configured sqlite_dir's parent (the
    `data/` root used elsewhere), falling back to the literal default path."""
    p = Path(path) if path else Path(_DEFAULT_PATH)
    if not p.exists():
        # sqlite_dir is 'data/seed'; the eval set lives under the same data root.
        p = Path(settings.sqlite_dir).parent / "eval" / "golden_questions.json"
    return json.loads(p.read_text())
