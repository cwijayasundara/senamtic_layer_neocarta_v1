import json

import pytest

from semantic_layer.graph.client import reset_graph
from semantic_layer.agent.doc_tools import search_documents


@pytest.mark.neo4j
@pytest.mark.openai
def test_search_documents_returns_relevant_chunks(neo4j_driver, require_openai):
    from semantic_layer.ingest.doc_loader import load_document
    from semantic_layer.ingest.embeddings import embed_chunks
    reset_graph(neo4j_driver)
    load_document(neo4j_driver, {
        "doc_id": "doc:t", "title": "t", "path": "/tmp/t.pdf", "num_pages": 1,
        "chunks": [
            {"chunk_id": "doc:t:chunk:0", "doc_id": "doc:t", "ordinal": 0,
             "text": "NVIDIA Data Center revenue grew on Blackwell demand."},
            {"chunk_id": "doc:t:chunk:1", "doc_id": "doc:t", "ordinal": 1,
             "text": "Gaming GPUs shipped to retail partners."},
        ],
    })
    embed_chunks(neo4j_driver)
    hits = json.loads(search_documents.invoke({"query": "data center revenue blackwell"}))
    assert len(hits) > 0
    assert hits[0]["doc_id"] == "doc:t"
    assert "revenue" in hits[0]["text"].lower()
