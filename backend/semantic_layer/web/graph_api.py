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
    """Renderable graph for the UI.

    Structured layer: source + table nodes; HAS_TABLE + table-level REFERENCES edges.
    Document layer: document -> chunk -> entity, the entity -> value bridge, and the
    value -> owning table link (HAS_VALUE) that ties the documents back into the
    structured catalog — so a PDF renders as a connected context graph, not a blob."""
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

    # Document -> Chunk: the actual PDF content, split into passages.
    chunk_rows = driver().execute_query(
        """
        MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
        RETURN c.id AS id, c.ordinal AS ordinal, c.text AS text, d.id AS doc_id
        ORDER BY d.id, c.ordinal
        """,
        database_=settings.neo4j_database,
    ).records
    for r in chunk_rows:
        nodes[r["id"]] = {"id": r["id"], "label": f"¶{r['ordinal']}", "kind": "chunk",
                          "source": "documents", "text": (r["text"] or "")[:280]}
        edges.append({"source": r["doc_id"], "target": r["id"], "type": "HAS_CHUNK"})

    # Chunk -> Entity: the entities extracted from each passage. We surface only
    # entities that carry structure — mentioned by 2+ passages OR bridged to the
    # catalog — so the document graph reads as a constellation, not a hairball of
    # hundreds of one-off mentions. Chunks still attach to their document regardless.
    ent_rows = driver().execute_query(
        """
        MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
        WITH e, collect(DISTINCT c.id) AS chunk_ids
        WHERE size(chunk_ids) >= 2 OR exists((e)-[:REFERS_TO]->(:Value))
        UNWIND chunk_ids AS chunk_id
        RETURN chunk_id, e.norm AS norm, e.name AS name, e.label AS label
        """,
        database_=settings.neo4j_database,
    ).records
    for r in ent_rows:
        eid = f"entity:{r['norm']}"
        nodes.setdefault(eid, {"id": eid, "label": r["name"], "kind": "entity",
                               "source": "documents", "entityType": r["label"]})
        edges.append({"source": r["chunk_id"], "target": eid, "type": "MENTIONS"})

    # Entity -> Value bridge, and Value -> owning Table, linking docs to the catalog.
    bridge_rows = driver().execute_query(
        """
        MATCH (e:Entity)-[:REFERS_TO]->(v:Value)
        OPTIONAL MATCH (t:Table)-[:HAS_COLUMN]->(:Column)-[:HAS_VALUE]->(v)
        RETURN e.norm AS enorm, v.norm AS vnorm, v.name AS vname,
               collect(DISTINCT t.id) AS tables
        """,
        database_=settings.neo4j_database,
    ).records
    for r in bridge_rows:
        vid = f"value:{r['vnorm']}"
        nodes.setdefault(vid, {"id": vid, "label": r["vname"], "kind": "value",
                               "source": "catalog"})
        edges.append({"source": f"entity:{r['enorm']}", "target": vid, "type": "REFERS_TO"})
        for tid in r["tables"]:
            if tid in nodes:  # tie the bridge into the existing table graph
                edges.append({"source": vid, "target": tid, "type": "HAS_VALUE"})

    return {"nodes": list(nodes.values()), "edges": edges}
