import pytest

from semantic_layer.config import settings
from semantic_layer.graph.client import reset_graph
from semantic_layer.ingest.embeddings import embed_chunks


@pytest.mark.neo4j
@pytest.mark.openai
def test_embed_chunks_sets_vectors(neo4j_driver, require_openai):
    reset_graph(neo4j_driver)
    with neo4j_driver.session(database=settings.neo4j_database) as s:
        s.run("CREATE (:Chunk {id:'c1', text:'NVIDIA Blackwell Data Center revenue'})")
    embed_chunks(neo4j_driver)
    with neo4j_driver.session(database=settings.neo4j_database) as s:
        dim = s.run("MATCH (c:Chunk {id:'c1'}) RETURN size(c.embedding) AS d").single()["d"]
    assert dim == settings.embedding_dimensions


@pytest.mark.neo4j
def test_embed_facts_sets_fake_vectors(neo4j_driver, monkeypatch):
    from semantic_layer.ingest.embeddings import embed_facts

    reset_graph(neo4j_driver)
    monkeypatch.setattr(settings, "fake_embeddings", True)
    with neo4j_driver.session(database=settings.neo4j_database) as session:
        session.run("CREATE (:Fact {id:'f1', text:'Blackwell / drove / growth'})")

    embed_facts(neo4j_driver)

    with neo4j_driver.session(database=settings.neo4j_database) as session:
        dim = session.run("MATCH (f:Fact {id:'f1'}) RETURN size(f.embedding) AS d").single()["d"]
    assert dim == settings.embedding_dimensions


def test_embed_facts_fake_path_sets_vectors_and_index(monkeypatch):
    from semantic_layer.ingest import embeddings

    class _Result:
        def __init__(self, rows):
            self._rows = rows

        def data(self):
            return self._rows

    class _Session:
        def __init__(self, driver):
            self.driver = driver

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def run(self, query, **params):
            self.driver.runs.append((query, params))
            if "MATCH (f:Fact) WHERE f.embedding IS NULL" in query:
                return _Result([{"id": "f1", "text": "Blackwell / drove / growth"}])
            return _Result([])

    class _Driver:
        def __init__(self):
            self.runs = []

        def session(self, database):
            assert database == settings.neo4j_database
            return _Session(self)

    driver = _Driver()
    monkeypatch.setattr(settings, "fake_embeddings", True)
    monkeypatch.setattr(settings, "embedding_dimensions", 3)
    monkeypatch.setattr(embeddings, "fake_vector", lambda text, dim: [float(len(text)), float(dim), 1.0])
    monkeypatch.setattr(embeddings, "get_openai_client", lambda: pytest.fail("OpenAI client was called"))

    embeddings.embed_facts(driver, batch=1)

    vector_write = next(params for query, params in driver.runs if "MATCH (f:Fact {id: row.id})" in query)
    assert vector_write["rows"] == [{"id": "f1", "vec": [26.0, 3.0, 1.0]}]
    assert any("CREATE VECTOR INDEX fact_embeddings IF NOT EXISTS" in query for query, _ in driver.runs)
