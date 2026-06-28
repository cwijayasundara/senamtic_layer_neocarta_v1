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


def get_schema_graph(source: str | None = None, max_chunks: int | None = None) -> dict:
    """Renderable graph for the UI, BOUNDED for scale.

    `source` (a Database name) restricts the structured layer to that source's tables;
    the document layer is included only when source is None or 'documents'. The chunk
    layer is capped at max_chunks (default settings.graph_max_chunks); entity/bridge
    edges are computed only over the included chunks. Returns {nodes, edges, truncated}."""
    cap = max_chunks if max_chunks is not None else settings.graph_max_chunks
    include_docs = source is None or source == "documents"
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    truncated = False

    db_filter = "WHERE d.name = $source" if (source and source != "documents") else ""
    db_rows = driver().execute_query(
        f"MATCH (d:Database) {db_filter} RETURN d.id AS id, d.name AS name, d.platform AS platform",
        source=source, database_=settings.neo4j_database,
    ).records
    for r in db_rows:
        platform = (r["platform"] or "").upper()
        nodes[r["id"]] = {"id": r["id"], "label": r["name"], "kind": "source",
                          "source": r["name"],
                          "platform": "sql" if platform in _SQL_PLATFORMS else "api"}

    tbl_filter = "WHERE d.name = $source" if (source and source != "documents") else ""
    tbl_rows = driver().execute_query(
        f"""
        MATCH (d:Database)-[:HAS_SCHEMA]->(:Schema)-[:HAS_TABLE]->(t:Table)
        {tbl_filter}
        RETURN t.id AS id, t.name AS name, d.id AS db_id, d.name AS source
        """,
        source=source, database_=settings.neo4j_database,
    ).records
    for r in tbl_rows:
        nodes[r["id"]] = {"id": r["id"], "label": r["name"], "kind": "table", "source": r["source"]}
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
        if r["a"] in nodes and r["b"] in nodes:   # only edges between included tables
            edges.append({"source": r["a"], "target": r["b"], "type": "REFERENCES"})

    if not include_docs:
        return {"nodes": list(nodes.values()), "edges": edges, "truncated": truncated}

    doc_rows = driver().execute_query(
        "MATCH (d:Document) RETURN d.id AS id, d.title AS title",
        database_=settings.neo4j_database,
    ).records
    for r in doc_rows:
        nodes[r["id"]] = {"id": r["id"], "label": r["title"], "kind": "document", "source": "documents"}

    chunk_rows = driver().execute_query(
        """
        MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
        RETURN c.id AS id, c.ordinal AS ordinal, c.text AS text, d.id AS doc_id
        ORDER BY d.id, c.ordinal
        LIMIT $cap_plus
        """,
        cap_plus=cap + 1, database_=settings.neo4j_database,
    ).records
    if len(chunk_rows) > cap:
        truncated = True
        chunk_rows = chunk_rows[:cap]
    chunk_ids = [r["id"] for r in chunk_rows]
    for r in chunk_rows:
        nodes[r["id"]] = {"id": r["id"], "label": f"¶{r['ordinal']}", "kind": "chunk",
                          "source": "documents", "text": (r["text"] or "")[:280]}
        edges.append({"source": r["doc_id"], "target": r["id"], "type": "HAS_CHUNK"})

    ent_rows = driver().execute_query(
        """
        MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
        WHERE c.id IN $chunk_ids
        OPTIONAL MATCH (e)-[:INSTANCE_OF]->(s:OntologySubtype)
        WITH e,
             collect(DISTINCT c.id) AS chunk_ids,
             [x IN collect(DISTINCT s.name) WHERE x IS NOT NULL] AS subtypes
        WHERE size(chunk_ids) >= 2 OR exists((e)-[:REFERS_TO]->(:Value)) OR size(subtypes) > 0
        UNWIND chunk_ids AS chunk_id
        RETURN chunk_id, e.norm AS norm, e.name AS name, e.label AS label, subtypes
        """,
        chunk_ids=chunk_ids, database_=settings.neo4j_database,
    ).records
    for r in ent_rows:
        eid = f"entity:{r['norm']}"
        # Multiple subtype edges can exist; expose one deterministic subtype.
        subtypes = sorted(r["subtypes"] or [])
        subtype = subtypes[0] if subtypes else None
        nodes.setdefault(eid, {"id": eid, "label": r["name"], "kind": "entity",
                               "source": "documents", "entityType": r["label"],
                               "subtype": subtype})
        edges.append({"source": r["chunk_id"], "target": eid, "type": "MENTIONS"})

    bridge_rows = driver().execute_query(
        """
        MATCH (e:Entity)-[:REFERS_TO]->(v:Value)
        OPTIONAL MATCH (t:Table)-[:HAS_COLUMN]->(:Column)-[:HAS_VALUE]->(v)
        RETURN e.norm AS enorm, v.norm AS vnorm, v.name AS vname, collect(DISTINCT t.id) AS tables
        """,
        database_=settings.neo4j_database,
    ).records
    for r in bridge_rows:
        eid = f"entity:{r['enorm']}"
        if eid not in nodes:        # only bridge entities that survived the chunk cap
            continue
        vid = f"value:{r['vnorm']}"
        nodes.setdefault(vid, {"id": vid, "label": r["vname"], "kind": "value", "source": "catalog"})
        edges.append({"source": eid, "target": vid, "type": "REFERS_TO"})
        for tid in r["tables"]:
            if tid in nodes:
                edges.append({"source": vid, "target": tid, "type": "HAS_VALUE"})

    return {"nodes": list(nodes.values()), "edges": edges, "truncated": truncated}
