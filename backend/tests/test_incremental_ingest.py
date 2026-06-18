import hashlib

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
