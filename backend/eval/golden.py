"""Load the golden NL-question set used to score the agent at scale."""

import json
from dataclasses import dataclass
from pathlib import Path

_DEFAULT = Path(__file__).with_name("golden_questions.json")


@dataclass
class GoldenQuestion:
    id: str
    question: str
    expected_tables: list[str]
    answer_check: dict
    category: str


def load_golden(path: str | None = None) -> list[GoldenQuestion]:
    data = json.loads(Path(path or _DEFAULT).read_text())
    return [GoldenQuestion(**row) for row in data]
