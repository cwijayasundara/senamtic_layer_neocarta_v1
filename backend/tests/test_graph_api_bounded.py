import pytest

from semantic_layer.web import graph_api


@pytest.mark.neo4j
def test_get_schema_graph_caps_chunks_and_sets_truncated(ingested_graph, monkeypatch):
    full = graph_api.get_schema_graph()
    assert "truncated" in full
    chunk_nodes = [n for n in full["nodes"] if n["kind"] == "chunk"]
    # With a tiny cap, the chunk layer is limited and truncated flips True.
    capped = graph_api.get_schema_graph(max_chunks=1)
    capped_chunks = [n for n in capped["nodes"] if n["kind"] == "chunk"]
    assert len(capped_chunks) <= 1
    if len(chunk_nodes) > 1:
        assert capped["truncated"] is True
    # Every edge references a node that is present (no dangling chunk/entity edges).
    ids = {n["id"] for n in capped["nodes"]}
    for e in capped["edges"]:
        assert e["source"] in ids and e["target"] in ids


@pytest.mark.neo4j
def test_get_schema_graph_source_filter_excludes_other_sources(ingested_graph):
    out = graph_api.get_schema_graph(source="sales_pg")
    sources = {n.get("source") for n in out["nodes"] if n["kind"] == "table"}
    assert sources <= {"sales_pg"}            # only sales_pg tables
    assert not [n for n in out["nodes"] if n["kind"] == "chunk"]  # docs excluded under a SQL source
