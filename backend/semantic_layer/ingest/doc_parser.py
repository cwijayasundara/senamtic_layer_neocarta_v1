"""Parse PDFs with liteparse v2 and split into overlapping chunks."""

import hashlib
from pathlib import Path

from liteparse import LiteParse


def file_content_hash(path: str) -> str:
    """sha256 hex of a file's bytes — identity key for incremental ingestion."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chunk_text(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        chunks.append(text[start : start + size])
        start += step
    return chunks


def parse_document(path: str, size: int = 1200, overlap: int = 150) -> dict:
    result = LiteParse().parse(path)
    doc_id = f"doc:{Path(path).stem}"
    pieces = chunk_text(result.text, size=size, overlap=overlap)
    chunks = [
        {"chunk_id": f"{doc_id}:chunk:{i}", "doc_id": doc_id, "ordinal": i, "text": piece}
        for i, piece in enumerate(pieces)
    ]
    return {
        "doc_id": doc_id,
        "title": Path(path).stem,
        "path": str(path),
        "num_pages": result.num_pages,
        "file_hash": file_content_hash(path),
        "chunks": chunks,
    }
