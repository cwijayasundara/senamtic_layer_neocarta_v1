import hashlib

import pytest

from semantic_layer.ingest.doc_parser import file_content_hash


def test_file_content_hash_is_stable_sha256(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert file_content_hash(str(p)) == expected
    assert file_content_hash(str(p)) == file_content_hash(str(p))  # deterministic


def test_file_content_hash_changes_with_content(tmp_path):
    a = tmp_path / "a.bin"; a.write_bytes(b"one")
    b = tmp_path / "b.bin"; b.write_bytes(b"two")
    assert file_content_hash(str(a)) != file_content_hash(str(b))


@pytest.mark.neo4j
def test_document_unchanged_detects_matching_hash(ingested_graph):
    from semantic_layer.ingest.doc_loader import document_unchanged, load_document

    doc_id = "doc:incremental_probe"
    load_document(ingested_graph, {
        "doc_id": doc_id, "title": "probe", "path": "/tmp/probe.pdf",
        "num_pages": 1, "file_hash": "abc123",
        "chunks": [{"chunk_id": f"{doc_id}:chunk:0", "doc_id": doc_id, "ordinal": 0, "text": "x"}],
    })
    # No embedding on the chunk yet -> not "unchanged" (would need re-embed).
    assert document_unchanged(ingested_graph, doc_id, "abc123") is False
    # Give the chunk an embedding, then a matching hash is "unchanged".
    with ingested_graph.session() as s:
        s.run("MATCH (c:Chunk {id:$id}) CALL db.create.setNodeVectorProperty(c,'embedding',$v)",
              id=f"{doc_id}:chunk:0", v=[0.1, 0.2, 0.3])
    assert document_unchanged(ingested_graph, doc_id, "abc123") is True
    assert document_unchanged(ingested_graph, doc_id, "different") is False
    # cleanup
    with ingested_graph.session() as s:
        s.run("MATCH (d:Document {id:$id}) OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c) DETACH DELETE d, c",
              id=doc_id)
