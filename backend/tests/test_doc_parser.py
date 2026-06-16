from pathlib import Path

from semantic_layer.ingest.doc_parser import parse_document, chunk_text

DOCS = Path(__file__).resolve().parents[2] / "docs"


def test_chunk_text_splits_with_overlap():
    chunks = chunk_text("abcdefghij", size=4, overlap=1)
    assert chunks[0] == "abcd"
    assert all(len(c) <= 4 for c in chunks)
    assert len(chunks) >= 3


def test_chunk_text_empty():
    assert chunk_text("   ") == []


def test_parse_real_pdf_returns_chunks():
    pdf = DOCS / "NVIDIAAn_2025.pdf"
    doc = parse_document(str(pdf))
    assert doc["doc_id"]
    assert doc["num_pages"] > 0
    assert len(doc["chunks"]) > 0
    assert any("NVIDIA" in c["text"] for c in doc["chunks"])
    # chunk ids are unique and ordinals are sequential
    ids = [c["chunk_id"] for c in doc["chunks"]]
    assert len(ids) == len(set(ids))
    assert [c["ordinal"] for c in doc["chunks"]] == list(range(len(doc["chunks"])))
