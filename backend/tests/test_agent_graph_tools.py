import json

import pytest

from semantic_layer.agent.graph_tools import list_sources, get_table_schema


@pytest.mark.neo4j
def test_list_sources_includes_db_and_api(ingested_graph):
    data = json.loads(list_sources.invoke({}))
    names = {s["name"] for s in data}
    assert {"sales_pg", "financials", "org", "crm", "itsm", "partner", "dgx"} <= names
    kinds = {s["name"]: s["kind"] for s in data}
    assert kinds["sales_pg"] == "sql"
    assert kinds["crm"] == "api"


@pytest.mark.neo4j
def test_get_table_schema_for_order_line(ingested_graph):
    schema = json.loads(get_table_schema.invoke({"table_id": "table:sales_pg.sales.order_line"}))
    assert schema["sql_reference"] == "sales.order_line"
    assert schema["source"] == "sales_pg"
    col_names = {c["name"] for c in schema["columns"]}
    assert {"order_id", "product_id", "amount"} <= col_names
    assert any(c["is_foreign_key"] for c in schema["columns"])
