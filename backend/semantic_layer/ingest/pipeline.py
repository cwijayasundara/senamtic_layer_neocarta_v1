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
from semantic_layer.ingest.doc_parser import parse_document
from semantic_layer.ingest.doc_loader import load_document


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

        docs_dir = Path(settings.docs_dir)
        pdfs = sorted(docs_dir.glob("*.pdf"))
        for pdf in pdfs:
            doc = parse_document(str(pdf))
            load_document(driver, doc)
        counts["documents"] = len(pdfs)

        if with_llm:
            _run_llm_stages(driver, bundles)
        return counts
    finally:
        driver.close()


def _run_llm_stages(driver, bundles) -> None:
    from semantic_layer.ingest.entities import extract_entities
    from semantic_layer.ingest.glossary import generate_business_terms, load_business_terms
    from semantic_layer.ingest.embeddings import embed_chunks, embed_metadata_nodes

    columns = [
        {"column_id": c.id, "name": c.name, "table": c.id.split(".")[-2]}
        for b in bundles for c in b.columns
    ][:60]
    if columns:
        load_business_terms(driver, generate_business_terms(columns))

    with driver.session(database=settings.neo4j_database) as session:
        chunk_rows = session.run(
            "MATCH (c:Chunk) RETURN c.id AS id, c.text AS text ORDER BY c.id LIMIT 40"
        ).data()
    for row in chunk_rows:
        for ent in extract_entities(row["text"]):
            with driver.session(database=settings.neo4j_database) as session:
                session.run(
                    """
                    MERGE (e:Entity {name: $name}) SET e.label = $label
                    WITH e
                    MATCH (c:Chunk {id: $chunk_id})
                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    name=ent["name"], label=ent["label"], chunk_id=row["id"],
                )

    embed_chunks(driver)
    embed_metadata_nodes(driver)


if __name__ == "__main__":
    print(run_ingest())
