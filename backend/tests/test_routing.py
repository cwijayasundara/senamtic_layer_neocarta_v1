import json
import pytest

from semantic_layer.agent import routing


def test_retrieve_falls_back_to_keyword_when_no_vector(monkeypatch):
    # No table embeddings -> vector hits empty -> keyword fallback (prior behavior).
    monkeypatch.setattr(routing, "_vector_table_hits", lambda q, k: {})
    hits = [
        {"kind": "value", "id": "c1", "name": "EMEA", "table_id": "table:sales_pg.sales.region", "score": 3},
        {"kind": "column", "id": "c2", "name": "segment", "table_id": "table:sales_pg.sales.segment", "score": 2},
        {"kind": "value", "id": "c3", "name": "EMEA", "table_id": "table:sales_pg.sales.region", "score": 1},
    ]
    monkeypatch.setattr(routing, "search_catalog", type("T", (), {
        "invoke": staticmethod(lambda _a: json.dumps(hits))})())
    out = routing.retrieve_candidate_tables("revenue by segment in EMEA", k_ret=20)
    ids = [c["table_id"] for c in out]
    assert ids == ["table:sales_pg.sales.region", "table:sales_pg.sales.segment"]
    assert out[0]["score"] == 4  # 3 + 1 for region


def test_retrieve_unions_vector_and_value_hits(monkeypatch):
    monkeypatch.setattr(routing, "_vector_table_hits",
                        lambda q, k: {"table:sales_pg.sales.customer": 0.9})
    monkeypatch.setattr(routing, "_keyword_value_hits",
                        lambda q: {"table:sales_pg.sales.region": 3.0})
    out = routing.retrieve_candidate_tables("customers in EMEA", k_ret=20)
    ids = {c["table_id"] for c in out}
    assert "table:sales_pg.sales.customer" in ids   # from vector
    assert "table:sales_pg.sales.region" in ids     # from value layer


def test_retrieve_includes_value_table_absent_from_vector(monkeypatch):
    monkeypatch.setattr(routing, "_vector_table_hits",
                        lambda q, k: {"table:scale.scale_hr.payroll": 0.5})
    monkeypatch.setattr(routing, "_keyword_value_hits",
                        lambda q: {"table:sales_pg.sales.region": 2.0})
    out = routing.retrieve_candidate_tables("how many in EMEA", k_ret=20)
    ids = {c["table_id"] for c in out}
    assert "table:sales_pg.sales.region" in ids


@pytest.mark.neo4j
def test_retrieve_candidate_tables_finds_real_dimensions(ingested_graph):
    out = routing.retrieve_candidate_tables("Data Center revenue from Cloud customers in EMEA", k_ret=20)
    ids = {c["table_id"] for c in out}
    # 'EMEA' is a region value, 'Cloud' an industry value, 'Data Center' a segment value.
    assert "table:sales_pg.sales.region" in ids
    assert any(t.endswith("industry") or t.endswith("segment") for t in ids)


from semantic_layer.agent import routing as routing_mod


class _FakeStructured:
    def __init__(self, value):
        self._value = value

    def invoke(self, _messages):
        return self._value


class _FakeModel:
    def __init__(self, value):
        self._value = value

    def with_structured_output(self, _schema, **_kwargs):
        return _FakeStructured(self._value)


def test_rank_tables_filters_by_threshold_and_limit(monkeypatch):
    candidates = [
        {"table_id": "table:sales_pg.sales.segment", "score": 5},
        {"table_id": "table:sales_pg.sales.region", "score": 4},
        {"table_id": "table:sales_pg.sales.industry", "score": 1},
    ]
    scores = routing_mod._TableScores(scores=[
        routing_mod._TableScore(table_id="table:sales_pg.sales.segment", score=5),
        routing_mod._TableScore(table_id="table:sales_pg.sales.region", score=4),
        routing_mod._TableScore(table_id="table:sales_pg.sales.industry", score=1),  # below min_score
    ])
    monkeypatch.setattr(routing_mod, "get_chat_model", lambda model=None: _FakeModel(scores))
    out = routing_mod.rank_tables("revenue by segment in EMEA", candidates, k_rank=8, min_score=3)
    assert out == ["table:sales_pg.sales.segment", "table:sales_pg.sales.region"]


def test_route_tables_returns_empty_without_candidates(monkeypatch):
    monkeypatch.setattr(routing_mod, "retrieve_candidate_tables", lambda q, k_ret=20: [])
    assert routing_mod.route_tables("nonsense xyzzy") == []


def test_keyword_value_hits_filters_non_table_ids(monkeypatch):
    recs = [
        {"table_id": "table:sales_pg.sales.region", "score": 2},
        {"table_id": "column:sales_pg.sales.region.name", "score": 9},  # not a table id
    ]
    result = type("R", (), {"records": recs})()
    fake_driver = type("D", (), {"execute_query": staticmethod(lambda *a, **k: result)})()
    monkeypatch.setattr(routing, "driver", lambda: fake_driver)
    out = routing._keyword_value_hits("revenue in EMEA")
    assert out == {"table:sales_pg.sales.region": 2.0}   # non-table id dropped


@pytest.mark.neo4j
@pytest.mark.openai
def test_vector_routing_finds_customer_table(ingested_graph):
    # Embed tables for real, then the semantic query that keyword retrieval missed.
    from semantic_layer.ingest.embeddings import embed_tables
    embed_tables(ingested_graph)
    out = routing.retrieve_candidate_tables("How many customers are there in total?", k_ret=20)
    ids = {c["table_id"] for c in out}
    assert "table:sales_pg.sales.customer" in ids
