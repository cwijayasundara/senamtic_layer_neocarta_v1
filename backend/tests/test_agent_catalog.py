import json

import pytest

from semantic_layer.agent.graph_tools import search_catalog


@pytest.mark.neo4j
def test_search_catalog_finds_revenue_columns(ingested_graph):
    hits = json.loads(search_catalog.invoke({"query": "revenue amount"}))
    assert len(hits) > 0
    assert any(h["kind"] == "column" and "sales_pg" in h["table_id"] for h in hits)


@pytest.mark.neo4j
def test_search_catalog_matches_table_names(ingested_graph):
    hits = json.loads(search_catalog.invoke({"query": "ticket"}))
    assert any("itsm" in h.get("table_id", "") or "itsm" in h.get("id", "") for h in hits)
