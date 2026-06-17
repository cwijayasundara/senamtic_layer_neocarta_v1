"""Run the full graph-ingestion pipeline idempotently.

Order: reset -> SQL metadata -> API metadata -> documents -> entities ->
glossary bridge -> embeddings.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from semantic_layer.config import settings
from semantic_layer.graph.client import get_driver, reset_graph
from semantic_layer.apis.app import app
from semantic_layer.ingest.sql_extractor import extract_postgres, extract_sqlite
from semantic_layer.ingest.api_extractor import extract_all_apis
from semantic_layer.ingest.metadata_loader import load_bundle
from semantic_layer.ingest.value_indexer import index_values
from semantic_layer.ingest.period_indexer import index_periods
from semantic_layer.ingest.bridge import bridge_sources
from semantic_layer.ingest.doc_parser import parse_document
from semantic_layer.ingest.doc_loader import load_document
from semantic_layer.ingest.doc_graph import extract_period, link_document_period


def _api_spec_getter():
    client = TestClient(app)
    return lambda prefix: client.get(f"{prefix}/openapi.json").json()


def run_ingest(*, with_llm: bool = True, reset: bool = True) -> dict:
    driver = get_driver()
    counts = {}
    try:
        if reset:
            reset_graph(driver)

        sqlite_dir = Path(settings.sqlite_dir)
        bundles = [
            extract_postgres(settings.postgres_dsn, source="sales_pg"),
            extract_sqlite(str(sqlite_dir / "financials.db"), source="financials"),
            extract_sqlite(str(sqlite_dir / "org.db"), source="org"),
        ]
        bundles += extract_all_apis(_api_spec_getter(), ("crm", "itsm", "partner", "dgx"))
        for b in bundles:
            load_bundle(driver, b)
        counts["sources"] = len(bundles)

        # Index dimension row-values as :Value nodes (pure SQL, always runs) so the
        # graph can route value-filtered questions and documents can bridge to them.
        counts["values"] = index_values(driver)
        # Turn fiscal_period rows into :Period nodes so documents can bridge to them
        # and SQL aggregations can be scoped to a document's reported quarter.
        counts["periods"] = index_periods(driver)
        # Link API key columns (account_id) to their SQL counterpart (customer_id) so
        # join-path planning can fold REST endpoints into cross-source queries.
        counts["bridges"] = bridge_sources(driver)

        docs_dir = Path(settings.docs_dir)
        pdfs = sorted(docs_dir.glob("*.pdf"))
        for pdf in pdfs:
            doc = parse_document(str(pdf))
            load_document(driver, doc)
            # Deterministic period extraction (regex) — runs without the LLM.
            link_document_period(driver, doc["doc_id"], extract_period(doc))
        counts["documents"] = len(pdfs)

        if with_llm:
            _run_llm_stages(driver, bundles)
        return counts
    finally:
        driver.close()


def _run_llm_stages(driver, bundles) -> None:
    from semantic_layer.ingest.entities import extract_entities
    from semantic_layer.ingest.glossary import generate_business_terms, load_business_terms
    from semantic_layer.ingest.doc_graph import load_entities, bridge_entities_to_values
    from semantic_layer.ingest.embeddings import embed_chunks, embed_metadata_nodes

    columns = [
        {"column_id": c.id, "name": c.name, "table": c.id.split(".")[-2]}
        for b in bundles for c in b.columns
    ][:60]
    if columns:
        load_business_terms(driver, generate_business_terms(columns))

    with driver.session(database=settings.neo4j_database) as session:
        chunk_rows = session.run(
            "MATCH (c:Chunk) RETURN c.id AS id, c.text AS text ORDER BY c.id"
        ).data()
    for row in chunk_rows:
        load_entities(driver, row["id"], extract_entities(row["text"]))
    # Bridge document entities to the canonical value layer (Entity -> Value).
    bridge_entities_to_values(driver)

    embed_chunks(driver)
    embed_metadata_nodes(driver)


if __name__ == "__main__":
    print(run_ingest())
