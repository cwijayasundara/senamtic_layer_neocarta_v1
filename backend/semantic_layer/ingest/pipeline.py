"""Run the full graph-ingestion pipeline idempotently.

Order: reset -> SQL metadata -> API metadata -> documents -> entities ->
glossary bridge -> embeddings.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from semantic_layer.config import settings
from semantic_layer.ingest.entities import extract_entities_batch
from semantic_layer.ingest.facts import extract_facts_batch
from semantic_layer.graph.client import get_driver, reset_graph
from semantic_layer.apis.app import app
from semantic_layer.ingest.sql_extractor import extract_postgres, extract_sqlite
from semantic_layer.ingest.api_extractor import extract_all_apis
from semantic_layer.ingest.synthetic_api import extract_synthetic_apis
from data.generators.scale_catalog import generate_scale_catalog
from semantic_layer.ingest.metadata_loader import load_bundle
from semantic_layer.ingest.value_indexer import index_values
from semantic_layer.ingest.period_indexer import index_periods
from semantic_layer.ingest.bridge import bridge_sources
from semantic_layer.ingest.query_log_indexer import index_query_log
from semantic_layer.ingest.doc_parser import parse_document, file_content_hash
from semantic_layer.ingest.doc_loader import load_document, document_unchanged
from semantic_layer.ingest.doc_graph import extract_period, link_document_period
from semantic_layer.ingest.ontology import load_ontology


def _api_spec_getter():
    client = TestClient(app)
    return lambda prefix: client.get(f"{prefix}/openapi.json").json()


def _scale_bundles() -> list:
    """Distractor SchemaBundles: one per scale_* schema (empty Postgres tables,
    introspected live) plus one per synthetic REST API. Empty when scale_mode off."""
    if not settings.scale_mode:
        return []
    catalog = generate_scale_catalog(
        seed=settings.random_seed,
        n_tables=settings.scale_n_tables,
        n_apis=settings.scale_n_apis,
    )
    schemas = sorted({t.schema for t in catalog.tables})
    pg_bundles = [
        extract_postgres(settings.postgres_dsn, source="scale", schema_name=s)
        for s in schemas
    ]
    if sum(len(b.tables) for b in pg_bundles) == 0:
        print(
            "WARNING: scale_mode is on but no scale_* distractor tables were found "
            "in Postgres. Run `make scale-seed` before `make scale-ingest`, or the "
            "scale test runs against an empty distractor catalog."
        )
    return pg_bundles + extract_synthetic_apis(catalog)


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
        bundles += extract_all_apis(_api_spec_getter(), settings.api_source_list)
        scale = _scale_bundles()
        bundles += scale
        counts["scale_sources"] = len(scale)
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
        # Mine the query log (if present) for empirically-observed joins via NeoCarta's
        # query_log connector, so join-path planning can use real usage, not just FKs.
        counts["observed_joins"] = index_query_log(driver)

        docs_dir = Path(settings.docs_dir)
        pdfs = sorted(docs_dir.glob("*.pdf"))
        ingested, skipped = 0, 0
        for pdf in pdfs:
            doc_id = f"doc:{pdf.stem}"
            if not reset and settings.ingest_skip_unchanged and \
                    document_unchanged(driver, doc_id, file_content_hash(str(pdf))):
                skipped += 1
                continue
            doc = parse_document(str(pdf))
            load_document(driver, doc)
            # Deterministic period extraction (regex) — runs without the LLM.
            link_document_period(driver, doc["doc_id"], extract_period(doc))
            ingested += 1
        counts["documents"] = ingested
        counts["documents_skipped"] = skipped
        counts["ontology_subtypes"] = load_ontology(driver)

        if with_llm:
            _run_llm_stages(driver, bundles)
        return counts
    finally:
        driver.close()


def extract_entities_for_chunks(chunk_rows: list[dict]) -> dict[str, list[dict]]:
    """Map chunk id -> entities, running entity_batch_size-sized batches concurrently.

    Replaces the previous one-LLM-call-per-chunk serial loop. Each batch is one LLM
    call; batches run across ingest_max_workers threads."""
    size = max(1, settings.entity_batch_size)
    batches = [chunk_rows[i:i + size] for i in range(0, len(chunk_rows), size)]
    if not batches:
        return {}

    def run(batch: list[dict]) -> dict[str, list[dict]]:
        groups = extract_entities_batch([r["text"] for r in batch])
        return {r["id"]: ents for r, ents in zip(batch, groups)}

    out: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=settings.ingest_max_workers) as pool:
        for partial in pool.map(run, batches):
            out.update(partial)
    return out


def extract_facts_for_chunks(chunk_rows: list[dict]) -> dict[str, list[dict]]:
    """Map chunk id -> facts, running entity_batch_size-sized batches concurrently."""
    size = max(1, settings.entity_batch_size)
    batches = [chunk_rows[i:i + size] for i in range(0, len(chunk_rows), size)]
    if not batches:
        return {}

    def run(batch: list[dict]) -> dict[str, list[dict]]:
        groups = extract_facts_batch([r["text"] for r in batch])
        return {r["id"]: facts for r, facts in zip(batch, groups)}

    out: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=settings.ingest_max_workers) as pool:
        for partial in pool.map(run, batches):
            out.update(partial)
    return out


def _run_llm_stages(driver, bundles) -> None:
    from semantic_layer.ingest.glossary import generate_business_terms, load_business_terms
    from semantic_layer.ingest.doc_graph import load_entities, bridge_entities_to_values
    from semantic_layer.ingest.facts import load_facts, link_facts
    from semantic_layer.ingest.embeddings import embed_chunks, embed_tables, embed_facts

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
    entities_by_chunk = extract_entities_for_chunks(chunk_rows)
    for chunk_id, ents in entities_by_chunk.items():
        load_entities(driver, chunk_id, ents)
    # Bridge document entities to the canonical value layer (Entity -> Value).
    bridge_entities_to_values(driver)
    facts_by_chunk = extract_facts_for_chunks(chunk_rows)
    for chunk_id, facts in facts_by_chunk.items():
        load_facts(driver, chunk_id, facts)
    link_facts(driver)

    embed_chunks(driver)
    embed_tables(driver)
    embed_facts(driver)


if __name__ == "__main__":
    print(run_ingest())
