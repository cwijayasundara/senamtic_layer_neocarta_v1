# backend/tests/test_controller_trace.py
from semantic_layer.agent import controller as ctrl
from semantic_layer.agent import cache as cache_mod
from semantic_layer.agent.planner import Intent


def test_answer_event_includes_per_leg_trace(monkeypatch):
    monkeypatch.setattr(ctrl, "query_cache", cache_mod.QueryCache(max_entries=10, ttl_seconds=1000))
    monkeypatch.setattr(ctrl.settings, "query_cache_enabled", False, raising=False)
    monkeypatch.setattr(ctrl, "extract_intent",
                        lambda q: Intent(needs_sql=True, needs_doc=True, doc_query="d"))
    monkeypatch.setattr(ctrl, "build_plan", lambda intent, question=None: {
        "highlight": [], "api_correlations": [],
        "sql_legs": [{"source": "sales_pg", "fact_table": "t", "join_targets": [],
                      "filters": [], "scope": {}}],
        "doc_leg": {"doc_query": "d", "candidate_doc_ids": [], "periods": []}})
    monkeypatch.setattr(ctrl, "run_sql_leg", lambda leg: {
        "source": "sales_pg", "sql": "SELECT 1", "columns": ["n"], "rows": [[1]],
        "row_count": 1, "error": None})
    monkeypatch.setattr(ctrl, "run_doc_leg", lambda q: {
        "answer": "a", "citations": [], "doc_texts": [], "error": None})
    monkeypatch.setattr(ctrl, "_synthesize", lambda *a, **k: "answer")
    monkeypatch.setattr(ctrl, "check_numeric_grounding", lambda *a, **k: [])

    answer = list(ctrl.answer_stream("q"))[-1]
    trace = answer["trace"]
    assert {t["name"] for t in trace} == {"sql:sales_pg", "doc:doc"}
    assert all(isinstance(t["duration_ms"], float) and t["duration_ms"] >= 0 for t in trace)
    assert all(t["ok"] is True for t in trace)
