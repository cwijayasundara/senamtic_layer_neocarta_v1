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

    def with_structured_output(self, _schema, **_kwargs):
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


from semantic_layer.agent.legs import run_api_leg


def test_run_api_leg_executes_planned_calls(monkeypatch):
    plan_calls = legs_mod._ApiCalls(calls=[
        legs_mod._ApiCall(source="itsm", path="/tickets", params={"status": "open"})])
    monkeypatch.setattr(legs_mod, "get_chat_model", lambda model=None: _FakeModel(plan_calls))
    # call_api is a LangChain tool: the leg invokes it via .invoke({...}).
    monkeypatch.setattr(legs_mod, "call_api", type("T", (), {
        "invoke": staticmethod(lambda _args: json.dumps(
            {"status": 200, "data": [{"id": 1}, {"id": 2}]}))})())
    out = run_api_leg(["open tickets"])
    assert out["error"] is None
    assert out["calls"][0]["source"] == "itsm"
    assert out["calls"][0]["path"] == "/tickets"
    assert out["calls"][0]["row_count"] == 2


from semantic_layer.agent.legs import run_doc_leg


def test_run_doc_leg_retrieves_and_answers(monkeypatch):
    monkeypatch.setattr(legs_mod, "search_documents", type("T", (), {
        "invoke": staticmethod(lambda args: json.dumps([
            {"chunk_id": "doc:x:chunk:2", "doc_id": "doc:x",
             "text": "Data Center revenue was a record $60.4 billion.", "score": 0.9}]))})())
    answer = legs_mod._DocAnswer(answer="Data Center revenue was $60.4 billion (doc:x).")
    monkeypatch.setattr(legs_mod, "get_chat_model", lambda model=None: _FakeModel(answer))
    out = run_doc_leg("what drove Data Center growth")
    assert out["error"] is None
    assert out["citations"][0]["doc_id"] == "doc:x"
    assert "60.4" in out["answer"]
    assert any("60.4" in t for t in out["doc_texts"])
