import pytest

from semantic_layer.agent import routing


def test_select_fact_table_empty_returns_none():
    assert routing.select_fact_table([]) is None


@pytest.mark.neo4j
def test_select_fact_table_picks_order_line(ingested_graph):
    # order_line is the fact (most FKs); region is a leaf dimension (no FKs out).
    routed = ["table:sales_pg.sales.region", "table:sales_pg.sales.order_line",
              "table:sales_pg.sales.customer"]
    assert routing.select_fact_table(routed) == "table:sales_pg.sales.order_line"
