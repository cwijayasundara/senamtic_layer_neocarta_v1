import json
import pytest

from semantic_layer.agent import legs as legs_mod
from semantic_layer.agent.legs import run_sql_leg


class _FakeStructured:
    def __init__(self, value):
        self._value = value

    def invoke(self, _messages):
        return self._value


class _FakeModel:
    def __init__(self, draft):
        self._draft = draft

    def with_structured_output(self, _schema):
        return _FakeStructured(self._draft)


def test_run_sql_leg_executes_drafted_sql(monkeypatch):
    leg = {"source": "financials", "fact_table": "table:financials.main.income_statement",
           "join_targets": [], "filters": [], "scope": {"fiscal_year": 2027, "quarter": "Q1"},
           "metrics": ["revenue"]}
    draft = legs_mod._SqlDraft(sql="SELECT revenue FROM income_statement LIMIT 1")
    monkeypatch.setattr(legs_mod, "get_chat_model", lambda model=None: _FakeModel(draft))
    captured = {}
    monkeypatch.setattr(legs_mod, "_run",
                        lambda source, sql: captured.update(source=source, sql=sql)
                        or json.dumps({"columns": ["revenue"], "rows": [[67795.0]]}))
    out = run_sql_leg(leg)
    assert captured["source"] == "financials"
    assert out["sql"].lower().startswith("select")
    assert out["columns"] == ["revenue"]
    assert out["row_count"] == 1
    assert out["error"] is None


def test_run_sql_leg_retries_once_on_error(monkeypatch):
    leg = {"source": "sales_pg", "fact_table": "table:sales_pg.sales.order_line",
           "join_targets": [], "filters": [], "scope": {"fiscal_year": None, "quarter": None}}
    draft = legs_mod._SqlDraft(sql="SELECT bad")
    monkeypatch.setattr(legs_mod, "get_chat_model", lambda model=None: _FakeModel(draft))
    calls = []
    monkeypatch.setattr(legs_mod, "_run",
                        lambda source, sql: calls.append(sql)
                        or json.dumps({"error": "boom"}))
    out = run_sql_leg(leg)
    assert len(calls) == 2          # initial + one retry
    assert out["error"] == "boom"
    assert out["rows"] == []
