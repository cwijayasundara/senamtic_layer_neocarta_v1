"""Add cross-source bridge edges so API tables join to SQL tables in the graph.

REST APIs expose keys (e.g. account_id) that denote the same entity as a SQL primary
key (sales.customer.customer_id) but carry no modeled foreign key. We add explicit
SAME_ENTITY edges from each API key column to its SQL counterpart so join-path planning
can fold API endpoints into cross-source queries. Deterministic; no LLM, no inference.
"""

from semantic_layer.config import settings

# Explicit key map: an API column with this name denotes the given SQL column id.
_BRIDGES = {
    "account_id": "col:sales_pg.sales.customer.customer_id",
}

_CYPHER = """
UNWIND $pairs AS pair
MATCH (db:Database)-[:HAS_SCHEMA]->(:Schema)-[:HAS_TABLE]->(:Table)-[:HAS_COLUMN]->(ac:Column)
  WHERE db.platform = 'REST-API' AND ac.name = pair.key
MATCH (sql:Column {id: pair.target})
MERGE (ac)-[:SAME_ENTITY]->(sql)
RETURN count(*) AS n
"""


def bridge_sources(driver) -> int:
    """MERGE SAME_ENTITY edges from API key columns to their SQL counterparts.

    Returns the number of (API column, SQL column) pairs linked. Idempotent."""
    pairs = [{"key": key, "target": target} for key, target in _BRIDGES.items()]
    records = driver.execute_query(
        _CYPHER, pairs=pairs, database_=settings.neo4j_database,
    ).records
    return records[0]["n"] if records else 0
