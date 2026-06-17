"""Graph-backed semantic tools: source catalog, schema lookup, join paths, search."""

import json
import re

from langchain_core.tools import tool

from semantic_layer.agent.driver import driver
from semantic_layer.agent.sql_tools import _run, _SQLITE_SOURCES
from semantic_layer.config import settings

_SQL_PLATFORMS = {"POSTGRESQL", "SQLITE"}
# Catalog-derived SQL identifiers (column name, schema-qualified table) must look
# like plain identifiers before they are ever interpolated into a query string.
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


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
def list_tables(source: str) -> str:
    """List every table in a source (e.g. 'sales_pg') with its table id and schema.

    Use this whenever a source is relevant but search_catalog did not surface all
    the tables you need. Question filters like 'EMEA', 'Cloud', 'Data Center', or
    'Blackwell' are ROW VALUES that live in lookup/dimension tables
    (region, industry, segment, architecture) — their table names never appear in
    the question, so keyword search misses them. Enumerate the source here, then use
    get_join_path to connect the dimension tables into the join. Returns a JSON array
    of {table_id, name, schema, source} ordered by name."""
    records = driver().execute_query(
        """
        MATCH (d:Database {name: $source})-[:HAS_SCHEMA]->(s:Schema)-[:HAS_TABLE]->(t:Table)
        RETURN t.id AS id, t.name AS name, s.name AS schema
        ORDER BY name
        """,
        source=source, database_=settings.neo4j_database,
    ).records
    out = [
        {"table_id": r["id"], "name": r["name"], "schema": r["schema"], "source": source}
        for r in records
    ]
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
    """Find the shortest join path between two tables (by id), across FK and cross-source bridges.

    Traverses REFERENCES edges in the graph and returns the ordered chain of
    tables plus the column pairs to JOIN on. Use this to build correct multi-table
    SQL — especially deep joins across many tables. Returns {found, tables, joins}."""
    # A table joins to itself with zero hops. shortestPath rejects a search where
    # start == end (Neo4j's forbid_shortestpath_common_nodes), so short-circuit here.
    if table_a_id == table_b_id:
        return json.dumps({"found": True, "tables": [table_a_id], "joins": []})
    # The join path alternates HAS_COLUMN (move into a table's column) and
    # REFERENCES (cross an FK to another table's column). A column-only path
    # would be disconnected, since columns within a table are not linked.
    try:
        records = driver().execute_query(
            """
            MATCH (ta:Table {id: $a}), (tb:Table {id: $b})
            MATCH p = shortestPath((ta)-[:HAS_COLUMN|REFERENCES|SAME_ENTITY*1..24]-(tb))
            RETURN [n IN nodes(p) | head(labels(n)) + '|' + n.id] AS nodes
            ORDER BY length(p) LIMIT 1
            """,
            a=table_a_id, b=table_b_id, database_=settings.neo4j_database,
        ).records
    except Exception as exc:  # noqa: BLE001 — surface graph errors to the agent, don't crash the run
        return json.dumps({"found": False, "tables": [], "joins": [], "error": str(exc)})
    if not records or not records[0]["nodes"]:
        return json.dumps({"found": False, "tables": [], "joins": []})
    nodes = records[0]["nodes"]
    tables = [n.split("|", 1)[1] for n in nodes if n.startswith("Table|")]
    cols = [n.split("|", 1)[1] for n in nodes if n.startswith("Column|")]
    # columns come in REFERENCES-linked pairs between adjacent tables
    joins = [{"on": [cols[i], cols[i + 1]]} for i in range(0, len(cols) - 1, 2)]
    return json.dumps({"found": True, "tables": tables, "joins": joins})


@tool
def resolve_value(value: str) -> str:
    """Find which SQL table/column a filter VALUE lives in, and its exact stored spelling.

    The catalog stores metadata, not rows, so you cannot tell from names alone that
    'Cloud' is an industry, 'Data Center' a segment, 'Blackwell' an architecture, or
    'EMEA' a region — and the stored string is often longer than the question's
    shorthand ('Cloud' -> 'Cloud Service Provider'). This samples the live 'name'
    columns of the SQL sources (case-insensitive substring match) and returns where the
    value lives. Call it once per filter term BEFORE planning a join, then filter on the
    returned table/column using the exact matched string. Returns a JSON array of
    {source, table_id, sql_reference, column, matches[]}."""
    name_cols = driver().execute_query(
        """
        MATCH (d:Database)-[:HAS_SCHEMA]->(:Schema)-[:HAS_TABLE]->(t:Table)-[:HAS_COLUMN]->(c:Column)
        WHERE toLower(c.name) = 'name' AND toUpper(coalesce(d.platform,'')) IN ['POSTGRESQL','SQLITE']
        RETURN d.name AS source, t.id AS table_id, c.name AS col
        ORDER BY table_id
        """,
        database_=settings.neo4j_database,
    ).records
    # The user value is bound as a parameter (never interpolated); identifiers come
    # from the trusted catalog but are still validated as a defense-in-depth measure.
    pattern = "%" + value.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    out = []
    for r in name_cols:
        ref = _sql_reference(r["table_id"])
        col = r["col"]
        if not (_IDENT.match(col) and _IDENT.match(ref)):
            continue
        placeholder = "?" if r["source"] in _SQLITE_SOURCES else "%s"
        sql = (f"SELECT DISTINCT {col} AS v FROM {ref} "
               f"WHERE lower({col}) LIKE {placeholder} ESCAPE '\\' LIMIT 5")
        res = json.loads(_run(r["source"], sql, params=(pattern,)))
        rows = res.get("rows") or []
        if rows:
            out.append({
                "source": r["source"], "table_id": r["table_id"],
                "sql_reference": ref, "column": col,
                "matches": [row[0] for row in rows],
            })
    return json.dumps(out)


@tool
def neighbors(name: str) -> str:
    """Show the cross-source neighborhood of a value or entity (e.g. 'Blackwell').

    Bridges structured data and documents: returns which SQL tables/columns CONTAIN
    this value, and which documents/chunks MENTION it. Use it to connect the two
    worlds — e.g. confirm that an architecture the press releases discuss is the same
    one you have sales rows for, then quantify it with the sql subagent. Returns JSON
    {name, catalog:[{table_id, column, value}], documents:[{doc_id, chunks}]}."""
    key = " ".join((name or "").lower().split())
    catalog = driver().execute_query(
        """
        MATCH (c:Column)-[:HAS_VALUE]->(v:Value {norm: $key})
        MATCH (t:Table)-[:HAS_COLUMN]->(c)
        RETURN DISTINCT t.id AS table_id, c.name AS column, v.name AS value
        ORDER BY table_id
        """,
        key=key, database_=settings.neo4j_database,
    ).records
    # Documents mention it via a bridged entity (Entity->Value) or a direct name match.
    documents = driver().execute_query(
        """
        MATCH (ch:Chunk)-[:MENTIONS]->(e:Entity)
        WHERE e.norm = $key OR (e)-[:REFERS_TO]->(:Value {norm: $key})
        RETURN ch.doc_id AS doc_id, count(DISTINCT ch) AS chunks
        ORDER BY chunks DESC
        """,
        key=key, database_=settings.neo4j_database,
    ).records
    return json.dumps({
        "name": name,
        "catalog": [dict(r) for r in catalog],
        "documents": [dict(r) for r in documents],
    })


@tool
def periods_for_documents(doc_ids: list[str]) -> str:
    """Return the fiscal period(s) the given documents report, to scope a SQL aggregation.

    When an answer combines press releases with sales data, the documents describe a
    specific fiscal quarter while order_line facts are all-time. Call this with the
    resolved doc ids to get each document's period, then tell the sql subagent to filter
    on it. Returns a JSON list of {doc_id, key, fiscal_year, quarter, sql_available};
    sql_available=true means matching sales rows exist and the period can be filtered."""
    records = driver().execute_query(
        """
        MATCH (d:Document)-[:COVERS_PERIOD]->(p:Period)
        WHERE d.id IN $ids
        RETURN d.id AS doc_id, p.key AS key, p.fiscal_year AS fiscal_year,
               p.quarter AS quarter, p.fiscal_period_id IS NOT NULL AS sql_available
        ORDER BY doc_id, key
        """,
        ids=doc_ids, database_=settings.neo4j_database,
    ).records
    return json.dumps([dict(r) for r in records])


@tool
def search_catalog(query: str, limit: int = 20) -> str:
    """Search the catalog for tables, columns, and business terms matching a query.

    Case-insensitive keyword match over names/descriptions across all sources
    (databases and APIs). Returns ranked JSON hits with their source and table so
    you can pick where to get the data. Start here to route a question."""
    terms = [t for t in query.lower().split() if len(t) > 2]
    if not terms:
        terms = [query.lower()]
    column_hits = driver().execute_query(
        """
        UNWIND $terms AS term
        MATCH (c:Column)<-[:HAS_COLUMN]-(t:Table)
        WHERE toLower(c.name) CONTAINS term
        WITH c, t, count(*) AS score
        RETURN 'column' AS kind, c.id AS id, c.name AS name,
               t.id AS table_id, score ORDER BY score DESC LIMIT $limit
        """,
        terms=terms, limit=limit, database_=settings.neo4j_database,
    ).records
    table_hits = driver().execute_query(
        """
        UNWIND $terms AS term
        MATCH (t:Table) WHERE toLower(t.name) CONTAINS term
        WITH t, count(*) AS score
        RETURN 'table' AS kind, t.id AS id, t.name AS name,
               t.id AS table_id, score ORDER BY score DESC LIMIT $limit
        """,
        terms=terms, limit=limit, database_=settings.neo4j_database,
    ).records
    term_hits = driver().execute_query(
        """
        UNWIND $terms AS term
        MATCH (col:Column)-[:TAGGED_WITH]->(bt:BusinessTerm)
        WHERE toLower(bt.name) CONTAINS term OR toLower(coalesce(bt.description,'')) CONTAINS term
        RETURN DISTINCT 'business_term' AS kind, bt.id AS id, bt.name AS name,
               col.id AS table_id, 1 AS score LIMIT $limit
        """,
        terms=terms, limit=limit, database_=settings.neo4j_database,
    ).records
    # Value hits route data-value filters ('EMEA', 'Blackwell', 'Data Center') to the
    # table/column that holds them, with the exact stored spelling — what keyword
    # matching over names alone cannot do.
    value_hits = driver().execute_query(
        """
        UNWIND $terms AS term
        MATCH (c:Column)-[:HAS_VALUE]->(v:Value)
        WHERE toLower(v.name) CONTAINS term
        MATCH (t:Table)-[:HAS_COLUMN]->(c)
        WITH c, t, v, count(*) AS score
        RETURN 'value' AS kind, c.id AS id, v.name AS name,
               t.id AS table_id, c.name AS column, score
        ORDER BY score DESC LIMIT $limit
        """,
        terms=terms, limit=limit, database_=settings.neo4j_database,
    ).records
    hits = [dict(r) for r in (list(column_hits) + list(table_hits)
                              + list(value_hits) + list(term_hits))]
    return json.dumps(hits[:limit])
