# backend/tests/test_embed_tables.py
import pytest

from semantic_layer.ingest import embeddings
from semantic_layer.ingest.embeddings import _table_embed_text


def test_table_embed_text_includes_name_and_columns():
    assert _table_embed_text("order_line", ["line_id", "amount"]) == \
        "order_line — columns: line_id, amount"


def test_table_embed_text_without_columns():
    assert _table_embed_text("region", []) == "region"


@pytest.mark.neo4j
@pytest.mark.openai
def test_embed_tables_writes_embeddings_and_index(ingested_graph):
    driver = ingested_graph
    embeddings.embed_tables(driver)
    from semantic_layer.config import settings
    with driver.session(database=settings.neo4j_database) as s:
        missing = s.run(
            "MATCH (t:Table) WHERE t.embedding IS NULL RETURN count(t) AS c"
        ).single()["c"]
        idx = s.run(
            "SHOW VECTOR INDEXES YIELD name WHERE name = 'table_embeddings' RETURN count(*) AS c"
        ).single()["c"]
    assert missing == 0
    assert idx == 1
