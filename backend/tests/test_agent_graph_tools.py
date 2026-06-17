import json

import pytest

from semantic_layer.agent.graph_tools import (
    list_sources, list_tables, resolve_value, neighbors, search_catalog,
    get_table_schema,
)


@pytest.mark.neo4j
def test_list_sources_includes_db_and_api(ingested_graph):
    data = json.loads(list_sources.invoke({}))
    names = {s["name"] for s in data}
    assert {"sales_pg", "financials", "org", "crm", "itsm", "partner", "dgx"} <= names
    kinds = {s["name"]: s["kind"] for s in data}
    assert kinds["sales_pg"] == "sql"
    assert kinds["crm"] == "api"


@pytest.mark.neo4j
def test_list_tables_surfaces_dimension_tables(ingested_graph):
    # Dimension/lookup tables (region, industry, segment, architecture) are only
    # reachable by enumerating the source — their names never appear in questions
    # that filter on their row values ("EMEA", "Cloud", "Data Center", "Blackwell").
    data = json.loads(list_tables.invoke({"source": "sales_pg"}))
    names = {t["name"] for t in data}
    assert {
        "region", "country", "industry", "customer", "segment", "architecture",
        "product_line", "product", "fiscal_period", "sales_order", "order_line",
    } <= names
    by_name = {t["name"]: t for t in data}
    assert by_name["segment"]["table_id"] == "table:sales_pg.sales.segment"
    assert by_name["segment"]["source"] == "sales_pg"


@pytest.mark.neo4j
def test_list_tables_unknown_source_is_empty(ingested_graph):
    assert json.loads(list_tables.invoke({"source": "nope"})) == []


@pytest.mark.neo4j
@pytest.mark.postgres
def test_resolve_value_maps_shorthand_to_dimension(ingested_graph):
    # 'Cloud' is shorthand stored as 'Cloud Service Provider' in sales.industry —
    # the agent can only map it correctly by resolving the live value.
    hits = json.loads(resolve_value.invoke({"value": "Cloud"}))
    industry = [h for h in hits if h["table_id"] == "table:sales_pg.sales.industry"]
    assert industry, f"expected an industry hit, got {hits}"
    assert any("Cloud Service Provider" in m for m in industry[0]["matches"])

    seg = json.loads(resolve_value.invoke({"value": "Data Center"}))
    assert any(h["table_id"] == "table:sales_pg.sales.segment" for h in seg)


@pytest.mark.neo4j
@pytest.mark.postgres
def test_resolve_value_unknown_is_empty(ingested_graph):
    assert json.loads(resolve_value.invoke({"value": "zzz-nonexistent"})) == []


@pytest.mark.neo4j
@pytest.mark.postgres
def test_resolve_value_injection_is_neutralized(ingested_graph):
    # The value is bound as a parameter, so SQL metacharacters are treated as a
    # literal search string — no error, no rows, no injection.
    for payload in ["x' OR '1'='1", "'); DROP TABLE sales.region; --", "100%"]:
        assert json.loads(resolve_value.invoke({"value": payload})) == []


@pytest.mark.neo4j
def test_search_catalog_routes_value_to_owning_table(ingested_graph):
    # The value layer lets a data value route to its table even though the table
    # name never appears in the query.
    def value_hit_for(term, table_id):
        hits = json.loads(search_catalog.invoke({"query": term}))
        return [h for h in hits if h.get("kind") == "value" and h["table_id"] == table_id]

    assert value_hit_for("Blackwell", "table:sales_pg.sales.architecture")
    assert value_hit_for("Data Center", "table:sales_pg.sales.segment")
    cloud = value_hit_for("Cloud", "table:sales_pg.sales.industry")
    assert cloud and cloud[0]["name"] == "Cloud Service Provider"  # exact stored spelling


@pytest.mark.neo4j
def test_neighbors_returns_catalog_location_of_value(ingested_graph):
    data = json.loads(neighbors.invoke({"name": "Blackwell"}))
    cat = {(c["table_id"], c["column"]) for c in data["catalog"]}
    assert ("table:sales_pg.sales.architecture", "name") in cat
    assert all("doc_id" in d and "chunks" in d for d in data["documents"])


@pytest.mark.neo4j
def test_neighbors_bridges_documents_to_catalog(neo4j_driver):
    # Seed a value + a document entity with the same norm, bridge them, and confirm
    # neighbors() surfaces the document alongside the catalog location.
    from semantic_layer.config import settings
    from semantic_layer.graph.client import reset_graph
    from semantic_layer.ingest.doc_loader import load_document
    from semantic_layer.ingest.doc_graph import load_entities, bridge_entities_to_values

    reset_graph(neo4j_driver)
    with neo4j_driver.session(database=settings.neo4j_database) as s:
        s.run(
            """
            MERGE (v:Value {norm:'blackwell'}) SET v.name='Blackwell'
            MERGE (t:Table {id:'table:sales_pg.sales.architecture'}) SET t.name='architecture'
            MERGE (c:Column {id:'col:sales_pg.sales.architecture.name'}) SET c.name='name'
            MERGE (t)-[:HAS_COLUMN]->(c)
            MERGE (c)-[:HAS_VALUE]->(v)
            """
        )
    load_document(neo4j_driver, {
        "doc_id": "doc:pr", "title": "pr", "path": "/tmp/pr.pdf", "num_pages": 1,
        "chunks": [{"chunk_id": "doc:pr:chunk:0", "doc_id": "doc:pr", "ordinal": 0,
                    "text": "Blackwell drove growth."}],
    })
    load_entities(neo4j_driver, "doc:pr:chunk:0", [{"name": "Blackwell", "label": "Object"}])
    assert bridge_entities_to_values(neo4j_driver) == 1

    data = json.loads(neighbors.invoke({"name": "Blackwell"}))
    assert ("table:sales_pg.sales.architecture", "name") in {
        (c["table_id"], c["column"]) for c in data["catalog"]
    }
    assert any(d["doc_id"] == "doc:pr" for d in data["documents"])


@pytest.mark.neo4j
def test_get_table_schema_for_order_line(ingested_graph):
    schema = json.loads(get_table_schema.invoke({"table_id": "table:sales_pg.sales.order_line"}))
    assert schema["sql_reference"] == "sales.order_line"
    assert schema["source"] == "sales_pg"
    col_names = {c["name"] for c in schema["columns"]}
    assert {"order_id", "product_id", "amount"} <= col_names
    assert any(c["is_foreign_key"] for c in schema["columns"])
