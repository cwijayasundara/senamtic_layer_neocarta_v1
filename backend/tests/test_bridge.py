import pytest

from semantic_layer.config import settings


@pytest.mark.neo4j
def test_account_id_bridges_to_customer_id(ingested_graph):
    # ingested_graph runs run_ingest, which now calls bridge_sources.
    from semantic_layer.agent.driver import driver
    rows = driver().execute_query(
        """
        MATCH (ac:Column)-[:SAME_ENTITY]->(c:Column {id:'col:sales_pg.sales.customer.customer_id'})
        RETURN ac.id AS api_col ORDER BY api_col
        """,
        database_=settings.neo4j_database,
    ).records
    api_cols = [r["api_col"] for r in rows]
    assert "col:itsm.api.GET /tickets.account_id" in api_cols
    assert "col:dgx.api.GET /usage.account_id" in api_cols
    assert all(c.endswith(".account_id") for c in api_cols)
