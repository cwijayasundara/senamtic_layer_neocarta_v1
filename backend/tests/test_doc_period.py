"""Tests for fiscal-period extraction from document text (no DB needed)."""

from semantic_layer.ingest.doc_graph import extract_period


def _doc(*texts: str) -> dict:
    return {"chunks": [{"text": t} for t in texts]}


def test_extracts_quarter_and_fiscal_year_from_summary_header():
    doc = _doc("NVIDIA Announces Financial Results\n\nQ1 Fiscal 2027 Summary\nRevenue $81,615")
    assert extract_period(doc) == {"fiscal_year": 2027, "quarter": "Q1"}


def test_normalizes_two_digit_fy():
    doc = _doc("Outlook for Q1 FY27 and beyond")
    assert extract_period(doc) == {"fiscal_year": 2027, "quarter": "Q1"}


def test_falls_back_to_word_quarter_plus_fiscal_year():
    doc = _doc("Data Center\nFirst-quarter revenue was $39.1 billion for fiscal 2026, up 73%.")
    assert extract_period(doc) == {"fiscal_year": 2026, "quarter": "Q1"}


def test_returns_none_when_no_period_present():
    doc = _doc("NVIDIA is the world leader in accelerated computing.")
    assert extract_period(doc) is None
