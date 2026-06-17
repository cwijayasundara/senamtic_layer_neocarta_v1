import pytest
from fastapi.testclient import TestClient

from semantic_layer.web.app import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.neo4j
def test_sources_endpoint(ingested_graph):
    data = client.get("/sources").json()
    names = {s["name"] for s in data}
    assert {"sales_pg", "crm"} <= names


@pytest.mark.neo4j
def test_graph_endpoint_returns_tables_and_refs(ingested_graph):
    g = client.get("/graph").json()
    assert "nodes" in g and "edges" in g
    kinds = {n["kind"] for n in g["nodes"]}
    assert {"source", "table"} <= kinds
    assert any(e["type"] == "REFERENCES" for e in g["edges"])
    assert any(n["kind"] == "document" for n in g["nodes"])


@pytest.mark.neo4j
def test_graph_endpoint_expands_documents_into_chunks(ingested_graph):
    # Documents must not be lone "blob" nodes — they expand into chunk passages.
    g = client.get("/graph").json()
    chunks = [n for n in g["nodes"] if n["kind"] == "chunk"]
    assert chunks, "expected chunk nodes in the graph"
    assert any(c.get("text") for c in chunks)  # carries a passage preview
    assert any(e["type"] == "HAS_CHUNK" for e in g["edges"])
    # every chunk is connected to its document (no orphan blobs)
    doc_ids = {n["id"] for n in g["nodes"] if n["kind"] == "document"}
    has_chunk = [e for e in g["edges"] if e["type"] == "HAS_CHUNK"]
    assert all(e["source"] in doc_ids for e in has_chunk)


@pytest.mark.neo4j
def test_graph_endpoint_bridges_documents_to_catalog(neo4j_driver):
    # With a bridged entity, the graph links a document entity -> value -> table,
    # so the PDF context graph connects into the structured catalog.
    from semantic_layer.config import settings
    from semantic_layer.graph.client import reset_graph
    from semantic_layer.ingest.pipeline import run_ingest
    from semantic_layer.ingest.doc_graph import load_entities, bridge_entities_to_values

    reset_graph(neo4j_driver)
    run_ingest(with_llm=False, reset=False)  # metadata + values + chunks (no LLM)
    with neo4j_driver.session(database=settings.neo4j_database) as s:
        chunk_id = s.run("MATCH (c:Chunk) RETURN c.id AS id ORDER BY c.id LIMIT 1").single()["id"]
    load_entities(neo4j_driver, chunk_id, [{"name": "Blackwell", "label": "Object"}])
    assert bridge_entities_to_values(neo4j_driver) >= 1

    g = client.get("/graph").json()
    values = [n for n in g["nodes"] if n["kind"] == "value" and n["label"] == "Blackwell"]
    assert values, "expected a bridged Value node"
    assert any(e["type"] == "REFERS_TO" for e in g["edges"])
    assert any(e["type"] == "HAS_VALUE"
               and e["target"] == "table:sales_pg.sales.architecture" for e in g["edges"])
