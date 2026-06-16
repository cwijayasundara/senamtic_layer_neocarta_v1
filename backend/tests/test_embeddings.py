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
