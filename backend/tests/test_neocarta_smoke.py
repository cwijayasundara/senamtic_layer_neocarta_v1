import pytest

from neocarta.data_model.rdbms import (
    Database, Schema, Table, Column, HasSchema, HasTable, HasColumn, References,
)
from neocarta.ingest.rdbms import Neo4jRDBMSLoader

from semantic_layer.config import settings
from semantic_layer.graph.client import reset_graph


@pytest.mark.neo4j
def test_loader_writes_schema_layer(neo4j_driver):
    reset_graph(neo4j_driver)
    loader = Neo4jRDBMSLoader(neo4j_driver, database_name=settings.neo4j_database)

    loader.load_database_nodes([Database(id="db:test", name="test")])
    loader.load_schema_nodes([Schema(id="schema:test.public", name="public")])
    loader.load_table_nodes([Table(id="table:test.public.t", name="t")])
    loader.load_column_nodes([
        Column(id="col:test.public.t.a", name="a", type="INTEGER",
               nullable=False, is_primary_key=True, is_foreign_key=False),
        Column(id="col:test.public.t.b", name="b", type="INTEGER",
               nullable=True, is_primary_key=False, is_foreign_key=True),
    ])
    loader.load_has_schema_relationships([HasSchema(database_id="db:test", schema_id="schema:test.public")])
    loader.load_has_table_relationships([HasTable(schema_id="schema:test.public", table_id="table:test.public.t")])
    loader.load_has_column_relationships([
        HasColumn(table_id="table:test.public.t", column_id="col:test.public.t.a"),
        HasColumn(table_id="table:test.public.t", column_id="col:test.public.t.b"),
    ])
    loader.load_references_relationships([
        References(source_column_id="col:test.public.t.b", target_column_id="col:test.public.t.a"),
    ])

    with neo4j_driver.session(database=settings.neo4j_database) as s:
        cols = s.run(
            "MATCH (t:Table {id:'table:test.public.t'})-[:HAS_COLUMN]->(c:Column) RETURN count(c) AS c"
        ).single()["c"]
        refs = s.run("MATCH (:Column)-[r:REFERENCES]->(:Column) RETURN count(r) AS c").single()["c"]
    assert cols == 2
    assert refs == 1
