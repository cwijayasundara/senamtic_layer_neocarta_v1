import pytest

from semantic_layer.config import settings
from semantic_layer.graph.client import reset_graph
from semantic_layer.ingest.sql_extractor import extract_sqlite
from semantic_layer.ingest.metadata_loader import load_bundle


@pytest.mark.neo4j
def test_load_sqlite_bundle_creates_nodes(neo4j_driver, tmp_path):
    from data.seed_sqlite import seed_all
    seed_all(out_dir=str(tmp_path))
    reset_graph(neo4j_driver)
    bundle = extract_sqlite(str(tmp_path / "org.db"), source="org")
    load_bundle(neo4j_driver, bundle)
    with neo4j_driver.session(database=settings.neo4j_database) as s:
        tables = s.run(
            "MATCH (:Database {id:'db:org'})-[:HAS_SCHEMA]->(:Schema)-[:HAS_TABLE]->(t:Table) "
            "RETURN count(t) AS c"
        ).single()["c"]
        fk = s.run("MATCH (:Column)-[r:REFERENCES]->(:Column) RETURN count(r) AS c").single()["c"]
    assert tables == 3
    assert fk >= 2
