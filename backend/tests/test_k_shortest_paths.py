import json

import pytest

from semantic_layer.agent.graph_tools import k_shortest_join_paths


def test_same_table_returns_zero_hop_path():
    out = json.loads(k_shortest_join_paths.invoke({
        "table_a_id": "table:sales_pg.sales.order_line",
        "table_b_id": "table:sales_pg.sales.order_line"}))
    assert out["found"] is True
    assert out["paths"][0]["tables"] == ["table:sales_pg.sales.order_line"]
    assert out["paths"][0]["joins"] == []


@pytest.mark.neo4j
def test_returns_ranked_paths_between_tables(ingested_graph):
    out = json.loads(k_shortest_join_paths.invoke({
        "table_a_id": "table:sales_pg.sales.order_line",
        "table_b_id": "table:sales_pg.sales.region", "k": 3}))
    assert out["found"] is True
    assert 1 <= len(out["paths"]) <= 3
    # Each path connects the two tables and carries an observed-weight score.
    for p in out["paths"]:
        assert p["tables"][0] == "table:sales_pg.sales.order_line"
        assert p["tables"][-1] == "table:sales_pg.sales.region"
        assert isinstance(p["observed"], int)
    # Ranked by observed weight DESC.
    weights = [p["observed"] for p in out["paths"]]
    assert weights == sorted(weights, reverse=True)
