"""Tests for the query_log connector wiring (NeoCarta query_log -> OBSERVED_JOIN)."""

from pathlib import Path

import pytest

from semantic_layer.config import settings
from semantic_layer.ingest.query_log_indexer import _observed_joins, index_query_log


def _log_path() -> str:
    # Tests run from backend/ (see Makefile/pytest pythonpath), so the configured
    # relative path resolves against that cwd just like the pipeline.
    return str(Path(settings.query_log_file))


def test_parser_extracts_expected_join_pairs():
    """The NeoCarta parser turns the sample SQL into the sales-schema join columns."""
    pairs, n_queries = _observed_joins(_log_path())
    assert n_queries >= 5
    # Direction-normalized keys: ((table, col), (table, col)) sorted.
    flat = {frozenset(k) for k in pairs}
    expected = [
        {("order_line", "order_id"), ("sales_order", "order_id")},
        {("sales_order", "customer_id"), ("customer", "customer_id")},
        {("customer", "country_id"), ("country", "country_id")},
        {("country", "region_id"), ("region", "region_id")},
        {("order_line", "product_id"), ("product", "product_id")},
        {("product", "product_line_id"), ("product_line", "product_line_id")},
        {("product_line", "segment_id"), ("segment", "segment_id")},
        {("product_line", "architecture_id"), ("architecture", "architecture_id")},
        {("sales_order", "fiscal_period_id"), ("fiscal_period", "fiscal_period_id")},
        {("customer", "industry_id"), ("industry", "industry_id")},
    ]
    for e in expected:
        assert e in flat, f"missing observed join {e}"
    # The region<-country<-customer<-sales_order<-order_line chain appears in two
    # queries, so its joins must be weighted above 1.
    region_join = pairs[tuple(sorted((("country", "region_id"), ("region", "region_id"))))]
    assert region_join >= 2

    # The cross-source join (financials.income_statement ⋈ org.headcount on
    # fiscal_year) has no FK anywhere — only the log reveals it.
    assert {("income_statement", "fiscal_year"), ("headcount", "fiscal_year")} in flat


@pytest.mark.neo4j
def test_observed_join_edges_land_on_canonical_columns(ingested_graph):
    from semantic_layer.agent.driver import driver

    # ingested_graph runs the pipeline, which now calls index_query_log.
    rows = driver().execute_query(
        """
        MATCH (a:Column)-[r:OBSERVED_JOIN]->(b:Column)
        RETURN count(r) AS edges, sum(r.observations) AS obs
        """,
        database_=settings.neo4j_database,
    ).records
    assert rows[0]["edges"] >= 8
    assert rows[0]["obs"] >= rows[0]["edges"]

    # Spot-check a specific empirically-observed FK join exists on canonical ids.
    hit = driver().execute_query(
        """
        MATCH (a:Column {id:'col:sales_pg.sales.order_line.order_id'})
              -[:OBSERVED_JOIN]-(b:Column {id:'col:sales_pg.sales.sales_order.order_id'})
        RETURN count(*) AS c
        """,
        database_=settings.neo4j_database,
    ).records
    assert hit[0]["c"] == 1


@pytest.mark.neo4j
def test_observed_join_discovers_fkless_cross_source_join(ingested_graph):
    """A join across two SQLite DBs with no FK exists ONLY because the log shows it."""
    from semantic_layer.agent.driver import driver

    inc = "col:financials.main.income_statement.fiscal_year"
    hc = "col:org.main.headcount.fiscal_year"

    # No declared FK connects these columns...
    fk = driver().execute_query(
        f"MATCH (:Column {{id:'{inc}'}})-[r:REFERENCES]-(:Column {{id:'{hc}'}}) RETURN count(r) AS c",
        database_=settings.neo4j_database,
    ).records[0]["c"]
    assert fk == 0
    # ...but the query log produced an OBSERVED_JOIN between them.
    oj = driver().execute_query(
        f"MATCH (:Column {{id:'{inc}'}})-[r:OBSERVED_JOIN]-(:Column {{id:'{hc}'}}) RETURN count(r) AS c",
        database_=settings.neo4j_database,
    ).records[0]["c"]
    assert oj == 1


@pytest.mark.neo4j
def test_get_join_path_finds_path_only_observed_join_enables(ingested_graph):
    """get_join_path now connects two FK-disconnected tables via the observed join."""
    import json

    from semantic_layer.agent.graph_tools import get_join_path

    path = json.loads(get_join_path.invoke({
        "table_a_id": "table:financials.main.income_statement",
        "table_b_id": "table:org.main.headcount",
    }))
    assert path["found"] is True
    assert "table:financials.main.income_statement" in path["tables"]
    assert "table:org.main.headcount" in path["tables"]
    joined_cols = {c for j in path["joins"] for c in j["on"]}
    assert "col:financials.main.income_statement.fiscal_year" in joined_cols
    assert "col:org.main.headcount.fiscal_year" in joined_cols


@pytest.mark.neo4j
def test_reingest_is_idempotent(ingested_graph):
    from semantic_layer.agent.driver import driver

    before = driver().execute_query(
        "MATCH ()-[r:OBSERVED_JOIN]->() RETURN count(r) AS c",
        database_=settings.neo4j_database,
    ).records[0]["c"]
    index_query_log(ingested_graph)  # run again without reset
    after = driver().execute_query(
        "MATCH ()-[r:OBSERVED_JOIN]->() RETURN count(r) AS c",
        database_=settings.neo4j_database,
    ).records[0]["c"]
    assert before == after
