"""Graph-backed semantic tools: source catalog, schema lookup, join paths, search."""

import json

from langchain_core.tools import tool

from semantic_layer.agent.driver import driver
from semantic_layer.config import settings

_SQL_PLATFORMS = {"POSTGRESQL", "SQLITE"}


def _sql_reference(table_id: str) -> str:
    # table:{source}.{schema}.{table} ; sqlite schema 'main' has no qualifier
    parts = table_id.split(":")[1].split(".")
    source, schema, table = parts[0], parts[1], ".".join(parts[2:])
    return table if schema == "main" else f"{schema}.{table}"


@tool
def list_sources() -> str:
    """List every data source in the semantic layer with its kind (sql or api).

    Returns a JSON array of {name, platform, kind}. Use this first to see what
    data exists before deciding how to answer a question."""
    rows = driver().execute_query(
        "MATCH (d:Database) RETURN d.name AS name, d.platform AS platform ORDER BY name",
        database_=settings.neo4j_database,
    ).records
    out = []
    for r in rows:
        platform = (r["platform"] or "").upper()
        out.append({
            "name": r["name"],
            "platform": platform,
            "kind": "sql" if platform in _SQL_PLATFORMS else "api",
        })
    return json.dumps(out)


@tool
def get_table_schema(table_id: str) -> str:
    """Get columns, types, keys, and the physical SQL reference for a table id.

    table_id looks like 'table:sales_pg.sales.order_line'. Returns JSON with
    source, sql_reference (use this in SQL), columns[], and foreign-key targets."""
    records = driver().execute_query(
        """
        MATCH (t:Table {id: $tid})-[:HAS_COLUMN]->(c:Column)
        OPTIONAL MATCH (c)-[:REFERENCES]->(rc:Column)
        RETURN c.name AS name, c.type AS type, c.is_primary_key AS pk,
               c.is_foreign_key AS fk, rc.id AS references
        ORDER BY name
        """,
        tid=table_id, database_=settings.neo4j_database,
    ).records
    if not records:
        return json.dumps({"error": f"table not found: {table_id}"})
    source = table_id.split(":")[1].split(".")[0]
    columns = [
        {"name": r["name"], "type": r["type"], "is_primary_key": r["pk"],
         "is_foreign_key": r["fk"], "references": r["references"]}
        for r in records
    ]
    return json.dumps({
        "table_id": table_id, "source": source,
        "sql_reference": _sql_reference(table_id), "columns": columns,
    })
