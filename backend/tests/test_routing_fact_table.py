import pytest

from semantic_layer.agent import routing


def test_select_fact_table_empty_returns_none():
    assert routing.select_fact_table([]) is None


def test_select_fact_table_no_sales_tables_returns_none():
    # Tables from other schemas must return None without touching the DB.
    result = routing.select_fact_table(
        ["table:other_db.x.y", "table:crm.api./accounts"]
    )
    assert result is None


@pytest.mark.neo4j
def test_select_fact_table_picks_order_line(ingested_graph):
    # order_line and customer BOTH have 2 direct FKs in this schema; depth2 breaks the tie.
    routed = ["table:sales_pg.sales.region", "table:sales_pg.sales.order_line",
              "table:sales_pg.sales.customer"]
    result = routing.select_fact_table(routed)
    assert result == "table:sales_pg.sales.order_line"
    # Explicitly confirm customer (equal direct FKs, alphabetically earlier) did not win.
    assert result != "table:sales_pg.sales.customer"
