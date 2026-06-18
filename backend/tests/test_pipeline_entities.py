from semantic_layer.ingest import pipeline as pipe


def test_extract_entities_for_chunks_covers_all_rows(monkeypatch):
    rows = [{"id": f"c{i}", "text": f"text {i}"} for i in range(25)]
    seen_batches = []

    def fake_batch(texts):
        seen_batches.append(len(texts))
        # one fake entity per chunk so we can assert mapping
        return [[{"name": t, "label": "Org"}] for t in texts]

    monkeypatch.setattr(pipe.settings, "entity_batch_size", 10, raising=False)
    monkeypatch.setattr(pipe.settings, "ingest_max_workers", 4, raising=False)
    monkeypatch.setattr(pipe, "extract_entities_batch", fake_batch)

    result = pipe.extract_entities_for_chunks(rows)
    assert set(result) == {f"c{i}" for i in range(25)}           # every chunk mapped
    assert result["c7"] == [{"name": "text 7", "label": "Org"}]  # correct row->entities
    assert sorted(seen_batches) == [5, 10, 10]                   # 25 rows -> 10+10+5
