import json

import pytest

from semantic_layer.agent.graph_tools import get_join_path


@pytest.mark.neo4j
def test_join_path_segment_to_region_is_deep(ingested_graph):
    result = json.loads(get_join_path.invoke({
        "table_a_id": "table:sales_pg.sales.segment",
        "table_b_id": "table:sales_pg.sales.region",
    }))
    assert result["found"] is True
    tables = result["tables"]
    assert tables[0] == "table:sales_pg.sales.segment"
    assert tables[-1] == "table:sales_pg.sales.region"
    assert len(tables) >= 6
    assert all("on" in hop for hop in result["joins"])


@pytest.mark.neo4j
def test_join_path_none_when_disconnected(ingested_graph):
    result = json.loads(get_join_path.invoke({
        "table_a_id": "table:sales_pg.sales.segment",
        "table_b_id": "table:financials.main.stock_price",
    }))
    assert result["found"] is False
