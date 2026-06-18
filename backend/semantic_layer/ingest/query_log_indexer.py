"""Mine real query logs for empirically-observed joins via NeoCarta's query_log connector.

The metadata layer's REFERENCES edges come purely from *declared* foreign keys
(sql_extractor.py). But a join that is actually run is the strongest signal for join-path
planning — and many real joins are never declared as FKs:
  * SQLite sources may omit FK constraints;
  * cross-source joins (e.g. financials.income_statement ⋈ org.headcount on fiscal_year)
    cannot be FKs at all — they span databases.
NeoCarta ships a `query_log` connector whose SQL parser turns a BigQuery-audit-log JSON
into discovered table/column joins; we reuse that *parser* and bridge its findings onto
our canonical catalog.

ID-scheme bridge: NeoCarta's parser emits dotted ids (`project.dataset.table`), while
our catalog uses prefixed ids (`table:source.schema.table`, `col:...`). Rather than load
NeoCarta's structural duplicates, we resolve its join columns to our canonical Columns by
(table name, column name) — across *all* SQL sources, so cross-source joins are found —
and write a weighted `(:Column)-[:OBSERVED_JOIN {observations}]->(:Column)` edge plus a
`:Query` provenance node. A name pair that resolves ambiguously (same table+column name in
two sources) is skipped rather than guessed.

This mirrors the pattern already used for APIs (api_extractor.py) and documents
(doc_graph.py): use NeoCarta/library machinery to extract, then bridge into our graph.
"""

from collections import Counter, defaultdict
from pathlib import Path

from neo4j import Driver

from neocarta.connectors.query_log.extract import QueryLogExtractor

from semantic_layer.config import settings


def _observed_joins(log_file: str) -> tuple[Counter, int]:
    """Parse the query log and tally canonical join pairs.

    Returns (counter keyed by a direction-normalized ((lt,lc),(rt,rc)) tuple ->
    number of logged queries that join on it, number of queries parsed)."""
    ext = QueryLogExtractor()
    # cache=False returns the raw (non-deduplicated) reference rows, so a join that
    # appears in N queries is counted N times — that count *is* the join's weight.
    result = ext.extract_info_from_query_log_json(log_file, "bigquery", cache=False)
    refs, queries = result["column_references_info"], result["query_info"]

    pairs: Counter = Counter()
    if refs is None or refs.empty:
        return pairs, len(queries)
    for _, row in refs.iterrows():
        left = (row["left_table_name"], row["left_column_name"])
        right = (row["right_table_name"], row["right_column_name"])
        if not all(left) or not all(right) or left == right:
            continue
        # Direction-normalize so a JOIN written either way collapses to one edge.
        pairs[tuple(sorted((left, right)))] += 1
    return pairs, len(queries)


def _canonical_column_index(session) -> dict[tuple[str, str], set[str]]:
    """Map (table name, column name) -> set of canonical Column ids, for SQL sources.

    A set (not a single id) so callers can detect and skip ambiguous name pairs."""
    rows = session.run(
        """
        MATCH (d:Database)-[:HAS_SCHEMA]->(:Schema)-[:HAS_TABLE]->(t:Table)-[:HAS_COLUMN]->(c:Column)
        WHERE toUpper(coalesce(d.platform,'')) IN ['POSTGRESQL','SQLITE']
        RETURN t.name AS table, c.name AS col, c.id AS id
        """
    ).data()
    index: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in rows:
        index[(r["table"], r["col"])].add(r["id"])
    return index


def index_query_log(driver: Driver, log_file: str | None = None) -> int:
    """Write OBSERVED_JOIN edges + a :Query provenance node from a query log. Idempotent.

    Resolves the parser's join columns to canonical Columns by table+column name across
    all SQL sources (cross-source joins included). Returns the number of OBSERVED_JOIN
    edges written (0 if the log is absent)."""
    path = Path(log_file or settings.query_log_file)
    if not path.exists():
        return 0

    pairs, n_queries = _observed_joins(str(path))
    with driver.session(database=settings.neo4j_database) as session:
        index = _canonical_column_index(session)

        # Resolve name pairs -> canonical id pairs, dropping unmatched/ambiguous ones.
        # Re-key by direction-normalized id pair so weights aggregate per real edge.
        edges: Counter = Counter()
        for (left, right), obs in pairs.items():
            lids, rids = index.get(left), index.get(right)
            if not lids or not rids or len(lids) != 1 or len(rids) != 1:
                continue  # unmatched, or ambiguous across sources — skip, don't guess
            a, b = sorted((next(iter(lids)), next(iter(rids))))
            if a != b:
                edges[(a, b)] += obs

        # Provenance: one :Query node records how many logged queries informed the joins.
        session.run(
            "MERGE (q:Query {id:'querylog'}) SET q.logged_queries = $n",
            n=n_queries,
        )
        for (a, b), obs in edges.items():
            session.run(
                """
                MATCH (a:Column {id:$a}), (b:Column {id:$b})
                MERGE (a)-[r:OBSERVED_JOIN]->(b)
                SET r.observations = $obs
                """,
                a=a, b=b, obs=obs,
            )
    return len(edges)
