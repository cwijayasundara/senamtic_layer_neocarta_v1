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


@pytest.mark.neo4j
def test_build_plan_adds_doc_context_api_keys_and_highlight(ingested_graph):
    intent = Intent(terms=["Blackwell", "Data Center"], fact="revenue",
                    needs_sql=True, needs_doc=True, needs_api=True,
                    doc_query="what drove Data Center growth",
                    api_intents=["dgx usage", "open tickets"])
    plan = build_plan(intent)

    # Documents that mention the entities are surfaced as candidates.
    assert plan["doc_leg"] is not None
    assert "doc:NVIDIAAn_2026" in plan["doc_leg"]["candidate_doc_ids"]
    assert plan["doc_leg"]["doc_query"] == "what drove Data Center growth"

    # API correlation keys come straight from the SAME_ENTITY bridge.
    pairs = {(k["sql_column"], k["api_column"]) for k in plan["api_correlations"]}
    assert ("col:sales_pg.sales.customer.customer_id",
            "col:itsm.api.GET /tickets.account_id") in pairs

    # Highlight is the union of plan node ids for the UI graph.
    assert "table:sales_pg.sales.segment" in plan["highlight"]
    assert "doc:NVIDIAAn_2026" in plan["highlight"]


@pytest.mark.neo4j
def test_build_plan_adds_financials_leg_and_scope(ingested_graph):
    intent = Intent(terms=["Blackwell"], needs_sql=True,
                    financial_metrics=["revenue", "gross margin"],
                    fiscal_year=2027, quarter="Q1")
    plan = build_plan(intent)
    fin = next((leg for leg in plan["sql_legs"] if leg["source"] == "financials"), None)
    assert fin is not None
    assert fin["fact_table"] == "table:financials.main.income_statement"
    assert fin["join_targets"] == [] and fin["filters"] == []
    assert fin["scope"] == {"fiscal_year": 2027, "quarter": "Q1"}
    assert all("scope" in leg for leg in plan["sql_legs"])


@pytest.mark.neo4j
def test_build_plan_aggregation_leg_from_group_by(ingested_graph):
    # A bare aggregation (no filter value) still produces a sales leg via group_by.
    intent = Intent(terms=[], needs_sql=True, fact="revenue", group_by=["segment"])
    plan = build_plan(intent)
    sales = next((l for l in plan["sql_legs"] if l["source"] == "sales_pg"), None)
    assert sales is not None
    targets = {jt["table_id"] for jt in sales["join_targets"]}
    assert "table:sales_pg.sales.segment" in targets
    assert sales["group_by"] == ["segment"]
    assert sales["filters"] == []
