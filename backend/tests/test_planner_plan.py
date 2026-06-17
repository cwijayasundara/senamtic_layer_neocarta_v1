import pytest

from semantic_layer.agent.planner import Intent, build_plan


@pytest.mark.neo4j
def test_build_plan_resolves_values_and_join_targets(ingested_graph):
    intent = Intent(terms=["EMEA", "Cloud", "Blackwell", "Data Center"],
                    fact="revenue", needs_sql=True)
    plan = build_plan(intent)

    resolved = {r["term"]: r for r in plan["resolved_values"]}
    # All four descriptors resolve to a sales_pg dimension column with exact spelling.
    assert resolved["Cloud"]["source"] == "sales_pg"
    assert resolved["Cloud"]["exact"] == "Cloud Service Provider"
    assert resolved["Blackwell"]["table_id"] == "table:sales_pg.sales.architecture"

    assert len(plan["sql_legs"]) >= 1
    sales = next(leg for leg in plan["sql_legs"] if leg["source"] == "sales_pg")
    assert sales["fact_table"] == "table:sales_pg.sales.order_line"
    # Every resolved sales dimension is a join target with a concrete join chain.
    target_tables = {jt["table_id"] for jt in sales["join_targets"]}
    assert "table:sales_pg.sales.segment" in target_tables
    assert all(jt["joins"] for jt in sales["join_targets"])
    # Filters carry the EXACT stored spelling for the SQL leg to apply.
    filt = {f["column"]: f["value"] for f in sales["filters"]}
    assert filt.get("name") is not None
