"""Deterministic groundedness guard: flag summary numbers absent from results.

A guard, not a proof. It catches fabricated or mis-transcribed figures, not
wrong-but-present ones, and is intentionally generous (wide tolerance + magnitude
and percent variants) to keep false positives low. Caveats are advisory and never
block an answer.
"""

import json
import re
from typing import Any

# A number with optional leading $, optional thousands separators, optional decimal,
# and an optional magnitude/percent suffix.
_NUM = re.compile(r"\$?\s?(\d[\d,]*(?:\.\d+)?)\s?([kKmMbB%])?")
_NOISE_YEARS = {2024.0, 2025.0, 2026.0, 2027.0}


def _variants(value: float, suffix: str) -> set[float]:
    """Expand one written number into the magnitudes it could mean."""
    out = {value}
    s = suffix.lower()
    if s == "k":
        out.add(value * 1e3)
    elif s == "m":
        out.add(value * 1e6)
    elif s == "b":
        out.add(value * 1e9)
    elif s == "%":
        out.add(value / 100.0)   # "38.81%" may be stored as 0.3881
    return out


def _numbers_in_text(text: str) -> set[float]:
    """Every number mention in free text, expanded to its magnitude variants."""
    found: set[float] = set()
    for raw, suffix in _NUM.findall(text or ""):
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        found |= _variants(value, suffix)
    return found


def _cell_numbers(cell: Any) -> set[float]:
    if isinstance(cell, bool):
        return set()
    if isinstance(cell, (int, float)):
        return {float(cell)}
    if isinstance(cell, str):
        return _numbers_in_text(cell)
    return set()


def _grounded_set(sql_runs: list, api_calls: list, doc_citations: list) -> set[float]:
    grounded: set[float] = set()
    for run in sql_runs:
        for row in run.get("rows", []) or []:
            for cell in row:
                grounded |= _cell_numbers(cell)
    blob = json.dumps([c.get("data") for c in api_calls], default=str)
    blob += " " + " ".join(c.get("quote", "") for c in doc_citations)
    grounded |= _numbers_in_text(blob)
    return grounded


def _matches(candidates: set[float], grounded: set[float]) -> bool:
    for a in candidates:
        for b in grounded:
            if abs(a - b) <= max(0.5, 0.01 * max(abs(a), abs(b))):
                return True
    return False


def _is_noise(value: float, suffix: str) -> bool:
    if not suffix and value.is_integer():
        if value in _NOISE_YEARS:      # calendar/fiscal years
            return True
        if abs(value) < 10:            # small ordinals/counts ("3 tables")
            return True
    return False


def check_numeric_grounding(content: str, sql_runs: list, api_calls: list,
                            doc_citations: list) -> list[str]:
    """Return one advisory caveat per summary number not found in any result."""
    grounded = _grounded_set(sql_runs, api_calls, doc_citations)
    caveats: list[str] = []
    seen: set[str] = set()
    for raw, suffix in _NUM.findall(content or ""):
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        if _is_noise(value, suffix):
            continue
        display = f"{raw}{suffix}"
        if display in seen:
            continue
        if not _matches(_variants(value, suffix), grounded):
            seen.add(display)
            caveats.append(
                f"The summary cites {display}, which was not found in any "
                f"retrieved result (SQL/API/document)."
            )
    return caveats
