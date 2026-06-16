import pytest

from semantic_layer.config import settings
from semantic_layer.graph.client import get_driver
from semantic_layer.ingest.pipeline import run_ingest


@pytest.mark.neo4j
@pytest.mark.postgres
def test_metadata_and_docs_ingest_without_llm(neo4j_driver):
    counts = run_ingest(with_llm=False, reset=True)
    assert counts["sources"] == 7
    assert counts["documents"] >= 1
    driver = get_driver()
    with driver.session(database=settings.neo4j_database) as s:
        tables = s.run("MATCH (t:Table) RETURN count(t) AS c").single()["c"]
        chunks = s.run("MATCH (c:Chunk) RETURN count(c) AS c").single()["c"]
        refs = s.run("MATCH (:Column)-[r:REFERENCES]->(:Column) RETURN count(r) AS c").single()["c"]
    driver.close()
    assert tables >= 11
    assert chunks > 0
    assert refs >= 10
