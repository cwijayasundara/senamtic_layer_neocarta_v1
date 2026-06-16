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


@tool
def get_join_path(table_a_id: str, table_b_id: str) -> str:
    """Find the shortest foreign-key join path between two tables (by id).

    Traverses REFERENCES edges in the graph and returns the ordered chain of
    tables plus the column pairs to JOIN on. Use this to build correct multi-table
    SQL — especially deep joins across many tables. Returns {found, tables, joins}."""
    # The join path alternates HAS_COLUMN (move into a table's column) and
    # REFERENCES (cross an FK to another table's column). A column-only path
    # would be disconnected, since columns within a table are not linked.
    records = driver().execute_query(
        """
        MATCH (ta:Table {id: $a}), (tb:Table {id: $b})
        MATCH p = shortestPath((ta)-[:HAS_COLUMN|REFERENCES*1..24]-(tb))
        RETURN [n IN nodes(p) | head(labels(n)) + '|' + n.id] AS nodes
        ORDER BY length(p) LIMIT 1
        """,
        a=table_a_id, b=table_b_id, database_=settings.neo4j_database,
    ).records
    if not records or not records[0]["nodes"]:
        return json.dumps({"found": False, "tables": [], "joins": []})
    nodes = records[0]["nodes"]
    tables = [n.split("|", 1)[1] for n in nodes if n.startswith("Table|")]
    cols = [n.split("|", 1)[1] for n in nodes if n.startswith("Column|")]
    # columns come in REFERENCES-linked pairs between adjacent tables
    joins = [{"on": [cols[i], cols[i + 1]]} for i in range(0, len(cols) - 1, 2)]
    return json.dumps({"found": True, "tables": tables, "joins": joins})
