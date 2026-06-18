import json
import pytest

from semantic_layer.agent import routing


def test_retrieve_candidate_tables_dedups_and_ranks(monkeypatch):
    # search_catalog is a LangChain tool invoked via .invoke({"query": ...}).
    hits = [
        {"kind": "value", "id": "c1", "name": "EMEA", "table_id": "table:sales_pg.sales.region", "score": 3},
        {"kind": "column", "id": "c2", "name": "segment", "table_id": "table:sales_pg.sales.segment", "score": 2},
        {"kind": "value", "id": "c3", "name": "EMEA", "table_id": "table:sales_pg.sales.region", "score": 1},
    ]
    monkeypatch.setattr(routing, "search_catalog", type("T", (), {
        "invoke": staticmethod(lambda _a: json.dumps(hits))})())
    out = routing.retrieve_candidate_tables("revenue by segment in EMEA", k_ret=20)
    ids = [c["table_id"] for c in out]
    assert ids == ["table:sales_pg.sales.region", "table:sales_pg.sales.segment"]  # deduped, score-summed, ranked
    assert out[0]["score"] == 4  # 3 + 1 for region
    assert len(out) <= 20


@pytest.mark.neo4j
def test_retrieve_candidate_tables_finds_real_dimensions(ingested_graph):
    out = routing.retrieve_candidate_tables("Data Center revenue from Cloud customers in EMEA", k_ret=20)
    ids = {c["table_id"] for c in out}
    # 'EMEA' is a region value, 'Cloud' an industry value, 'Data Center' a segment value.
    assert "table:sales_pg.sales.region" in ids
    assert any(t.endswith("industry") or t.endswith("segment") for t in ids)
