"""Read-only graph projections for the web UI (schema-level, renderable)."""

from semantic_layer.agent.driver import driver
from semantic_layer.config import settings

_SQL_PLATFORMS = {"POSTGRESQL", "SQLITE"}


def get_sources() -> list[dict]:
    rows = driver().execute_query(
        "MATCH (d:Database) RETURN d.name AS name, d.platform AS platform ORDER BY name",
        database_=settings.neo4j_database,
    ).records
    out = []
    for r in rows:
        platform = (r["platform"] or "").upper()
        out.append({"name": r["name"], "platform": platform,
                    "kind": "sql" if platform in _SQL_PLATFORMS else "api"})
    return out


def get_schema_graph() -> dict:
    """Source + table + document nodes; HAS_TABLE + table-level REFERENCES edges."""
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    db_rows = driver().execute_query(
        "MATCH (d:Database) RETURN d.id AS id, d.name AS name, d.platform AS platform",
        database_=settings.neo4j_database,
    ).records
    for r in db_rows:
        platform = (r["platform"] or "").upper()
        nodes[r["id"]] = {"id": r["id"], "label": r["name"], "kind": "source",
                          "source": r["name"],
                          "platform": "sql" if platform in _SQL_PLATFORMS else "api"}

    tbl_rows = driver().execute_query(
        """
        MATCH (d:Database)-[:HAS_SCHEMA]->(:Schema)-[:HAS_TABLE]->(t:Table)
        RETURN t.id AS id, t.name AS name, d.id AS db_id, d.name AS source
        """,
        database_=settings.neo4j_database,
    ).records
    for r in tbl_rows:
        nodes[r["id"]] = {"id": r["id"], "label": r["name"], "kind": "table",
                          "source": r["source"]}
        edges.append({"source": r["db_id"], "target": r["id"], "type": "HAS_TABLE"})

    ref_rows = driver().execute_query(
        """
        MATCH (t1:Table)-[:HAS_COLUMN]->(:Column)-[:REFERENCES]->(:Column)<-[:HAS_COLUMN]-(t2:Table)
        WHERE t1 <> t2
        RETURN DISTINCT t1.id AS a, t2.id AS b
        """,
        database_=settings.neo4j_database,
    ).records
    for r in ref_rows:
        edges.append({"source": r["a"], "target": r["b"], "type": "REFERENCES"})

    doc_rows = driver().execute_query(
        "MATCH (d:Document) RETURN d.id AS id, d.title AS title",
        database_=settings.neo4j_database,
    ).records
    for r in doc_rows:
        nodes[r["id"]] = {"id": r["id"], "label": r["title"], "kind": "document",
                          "source": "documents"}

    return {"nodes": list(nodes.values()), "edges": edges}
