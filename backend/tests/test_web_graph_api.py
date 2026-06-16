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
