"""Pure scoring functions: routing precision/recall and answer-assertion checks."""

import re


def routing_scores(routed: list[str], expected: list[str]) -> dict:
    """Precision/recall of routed table ids vs expected. Empty `expected` means the
    question does not assert a table set (doc-RAG / near-miss) -> treated as a pass."""
    if not expected:
        return {"precision": 1.0, "recall": 1.0, "hit": True}
    routed_set, expected_set = set(routed), set(expected)
    inter = routed_set & expected_set
    precision = len(inter) / len(routed_set) if routed_set else 0.0
    recall = len(inter) / len(expected_set)
    return {"precision": precision, "recall": recall, "hit": expected_set <= routed_set}


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def check_answer(answer: str, check: dict) -> bool:
    kind = check.get("type")
    if kind == "contains":
        low = answer.lower()
        return all(v.lower() in low for v in check.get("values", []))
    if kind == "numeric":
        target, tol = float(check["value"]), float(check.get("tol", 0))
        for m in _NUM.findall(answer):
            try:
                if abs(float(m.replace(",", "")) - target) <= tol:
                    return True
            except ValueError:
                continue
        return False
    return False
