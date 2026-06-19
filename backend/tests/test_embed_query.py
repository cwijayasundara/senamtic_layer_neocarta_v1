from semantic_layer.ingest import embeddings


class _FakeEmbeddings:
    def create(self, model, input, dimensions):
        # one vector per input string; deterministic, no network
        data = [type("E", (), {"embedding": [float(len(s))] * dimensions})() for s in input]
        return type("R", (), {"data": data})()


class _FakeClient:
    embeddings = _FakeEmbeddings()


def test_embed_query_returns_single_vector(monkeypatch):
    monkeypatch.setattr(embeddings, "get_openai_client", lambda: _FakeClient())
    monkeypatch.setattr(embeddings.settings, "embedding_dimensions", 4)
    out = embeddings.embed_query("hello")
    assert out == [5.0, 5.0, 5.0, 5.0]   # len("hello") == 5, dim 4
