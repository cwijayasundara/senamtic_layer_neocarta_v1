from semantic_layer.ingest.embeddings import fake_vector


def test_fake_vector_is_deterministic_and_sized():
    a = fake_vector("total revenue by region", 1536)
    b = fake_vector("total revenue by region", 1536)
    assert a == b
    assert len(a) == 1536
    assert all(isinstance(x, float) for x in a)


def test_fake_vector_differs_by_text():
    assert fake_vector("alpha", 64) != fake_vector("beta", 64)
