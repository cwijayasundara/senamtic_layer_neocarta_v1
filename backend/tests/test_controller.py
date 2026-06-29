from semantic_layer.agent import controller as ctrl
from semantic_layer.agent.planner import Intent


def _fake_synth(*_a, **_k):
    return "Segment Data Center leads; per doc:x revenue was $60.4 billion."


def test_answer_stream_runs_stages_and_emits_answer(monkeypatch):
    monkeypatch.setattr(ctrl, "extract_intent",
                        lambda q: Intent(terms=["Data Center"], needs_sql=True, needs_doc=True,
                                         doc_query="dc growth"))
    monkeypatch.setattr(ctrl, "build_plan", lambda intent, **kwargs: {
        "resolved_values": [], "highlight": ["table:sales_pg.sales.segment", "doc:x"],
        "sql_legs": [{"source": "sales_pg", "fact_table": "table:sales_pg.sales.order_line",
                      "join_targets": [], "filters": [], "scope": {}}],
        "doc_leg": {"doc_query": "dc growth", "candidate_doc_ids": ["doc:x"], "periods": []},
        "api_correlations": [],
        "routed_tables": [],
    })
    monkeypatch.setattr(ctrl, "run_sql_leg", lambda leg: {
        "source": "sales_pg", "sql": "SELECT 1", "columns": ["n"], "rows": [[60400000000]],
        "row_count": 1, "error": None})
    monkeypatch.setattr(ctrl, "run_doc_leg", lambda q: {
        "answer": "doc says $60.4 billion", "citations": [
            {"doc_id": "doc:x", "chunk_id": "doc:x:chunk:2", "quote": "$60.4 billion", "score": 0.9}],
        "doc_texts": ["Data Center revenue was a record $60.4 billion."], "error": None})
    monkeypatch.setattr(ctrl, "_synthesize", _fake_synth)

    events = list(ctrl.answer_stream("what drove Data Center growth?"))
    answer = events[-1]
    assert answer["type"] == "answer"
    assert answer["sql_runs"][0]["row_count"] == 1
    assert answer["doc_citations"][0]["doc_id"] == "doc:x"
    assert "table:sales_pg.sales.segment" in answer["highlight"]
    assert answer["caveats"] == []                       # $60.4B grounded by doc_texts
    # bounded: exactly one answer; leg/plan progress surfaced as tool_result events
    kinds = [e["type"] for e in events]
    assert kinds.count("answer") == 1
    assert "tool_result" in kinds


def test_answer_stream_augments_highlight_from_leg_results(monkeypatch):
    monkeypatch.setattr(ctrl.settings, "query_cache_enabled", False, raising=False)
    monkeypatch.setattr(ctrl, "extract_intent",
                        lambda q: Intent(needs_sql=False, needs_api=True, needs_doc=True,
                                         doc_query="Blackwell architecture",
                                         api_intents=["open tickets"]))
    monkeypatch.setattr(ctrl, "build_plan", lambda intent, **kwargs: {
        "resolved_values": [], "highlight": [],
        "sql_legs": [],
        "doc_leg": {"doc_query": "Blackwell architecture", "candidate_doc_ids": [], "periods": []},
        "api_correlations": [
            {
                "sql_column": "col:sales_pg.sales.customer.customer_id",
                "api_column": "col:itsm.api.GET /tickets.account_id",
            }
        ],
        "routed_tables": [],
    })
    monkeypatch.setattr(ctrl, "run_api_leg", lambda intents: {"calls": [
        {"source": "dgx", "path": "/usage", "params": {}, "status": 200,
         "row_count": 1, "data": [{"gpu_hours": 100}]}
    ]})
    monkeypatch.setattr(ctrl, "run_doc_leg", lambda q: {
        "answer": "doc says Blackwell is a product architecture",
        "citations": [
            {"doc_id": "doc:x", "chunk_id": "doc:x:chunk:2", "quote": "Blackwell", "score": 0.9}],
        "doc_texts": ["Blackwell is a product architecture."], "error": None})
    monkeypatch.setattr(ctrl, "_synthesize", lambda *a, **k: "answer")
    monkeypatch.setattr(ctrl, "check_numeric_grounding", lambda *a, **k: [])

    answer = list(ctrl.answer_stream("q"))[-1]

    assert "table:itsm.api.GET /tickets" in answer["highlight"]
    assert "table:dgx.api.GET /usage" in answer["highlight"]
    assert "doc:x" in answer["highlight"]
    assert "doc:x:chunk:2" in answer["highlight"]
