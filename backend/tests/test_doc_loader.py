import pytest

from semantic_layer.config import settings
from semantic_layer.graph.client import reset_graph
from semantic_layer.ingest.doc_loader import load_document


@pytest.mark.neo4j
def test_load_document_creates_doc_and_chunks(neo4j_driver):
    reset_graph(neo4j_driver)
    doc = {
        "doc_id": "doc:sample", "title": "Sample", "path": "/tmp/sample.pdf", "num_pages": 1,
        "chunks": [
            {"chunk_id": "doc:sample:chunk:0", "doc_id": "doc:sample", "ordinal": 0, "text": "Blackwell GPU"},
            {"chunk_id": "doc:sample:chunk:1", "doc_id": "doc:sample", "ordinal": 1, "text": "Data Center revenue"},
        ],
    }
    load_document(neo4j_driver, doc)
    with neo4j_driver.session(database=settings.neo4j_database) as s:
        n = s.run("MATCH (:Document {id:'doc:sample'})-[:HAS_CHUNK]->(c:Chunk) RETURN count(c) AS c").single()["c"]
    assert n == 2
    load_document(neo4j_driver, doc)  # idempotent
    with neo4j_driver.session(database=settings.neo4j_database) as s:
        n2 = s.run("MATCH (:Document {id:'doc:sample'})-[:HAS_CHUNK]->(c:Chunk) RETURN count(c) AS c").single()["c"]
    assert n2 == 2
