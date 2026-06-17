"""Index low-cardinality dimension VALUES into the graph as :Value nodes.

The catalog stores metadata, not rows — so a question filtered by a data value
('Blackwell', 'Data Center', 'EMEA') has no graph anchor. This stage turns the
distinct values of small text columns into canonical (:Value {name, norm}) nodes
linked Column-[:HAS_VALUE]->Value, so search_catalog can route on them and document
entities can bridge to them (Entity-[:REFERS_TO]->Value). Pure read-only SQL — runs
on every ingest, with or without the LLM stages.
"""

import json
import re

from neo4j import Driver

from semantic_layer.agent.sql_tools import _run
from semantic_layer.config import settings

# Catalog-derived identifiers must look like plain identifiers before interpolation.
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
# SQL text/character types whose distinct values are worth indexing.
_TEXT = re.compile(r"char|text|string|clob", re.IGNORECASE)
MAX_DISTINCT = 50


def norm(s: str) -> str:
    """Canonical matching key: lowercased, whitespace-collapsed."""
    return " ".join((s or "").lower().split())


def _sql_reference(table_id: str) -> str:
    # table:{source}.{schema}.{table}; sqlite schema 'main' has no qualifier.
    parts = table_id.split(":")[1].split(".")
    _, schema, table = parts[0], parts[1], ".".join(parts[2:])
    return table if schema == "main" else f"{schema}.{table}"


def index_values(driver: Driver, max_distinct: int = MAX_DISTINCT) -> int:
    """MERGE :Value nodes for low-cardinality text columns and link HAS_VALUE.

    Returns the number of Column-[:HAS_VALUE]->Value links written."""
    with driver.session(database=settings.neo4j_database) as session:
        cols = session.run(
            """
            MATCH (d:Database)-[:HAS_SCHEMA]->(:Schema)-[:HAS_TABLE]->(t:Table)-[:HAS_COLUMN]->(c:Column)
            WHERE toUpper(coalesce(d.platform,'')) IN ['POSTGRESQL','SQLITE']
            RETURN d.name AS source, t.id AS table_id,
                   c.id AS col_id, c.name AS col, c.type AS type
            ORDER BY col_id
            """
        ).data()

    linked = 0
    for r in cols:
        if not _TEXT.search(r["type"] or ""):
            continue
        ref, col = _sql_reference(r["table_id"]), r["col"]
        if not (_IDENT.match(ref) and _IDENT.match(col)):
            continue
        sql = (f"SELECT DISTINCT {col} AS v FROM {ref} "
               f"WHERE {col} IS NOT NULL LIMIT {max_distinct + 1}")
        res = json.loads(_run(r["source"], sql))
        rows = res.get("rows")
        if rows is None or len(rows) > max_distinct:
            continue  # SQL error, or high-cardinality column — skip
        values = sorted({str(row[0]).strip() for row in rows if str(row[0]).strip()})
        if not values:
            continue
        with driver.session(database=settings.neo4j_database) as session:
            session.run(
                """
                MATCH (c:Column {id: $col_id})
                UNWIND $values AS val
                MERGE (v:Value {norm: val.norm})
                  ON CREATE SET v.name = val.name
                MERGE (c)-[:HAS_VALUE]->(v)
                """,
                col_id=r["col_id"],
                values=[{"name": v, "norm": norm(v)} for v in values],
            )
        linked += len(values)
    return linked
