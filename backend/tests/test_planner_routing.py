# backend/tests/test_planner_routing.py
from semantic_layer.agent import planner as planner_mod
from semantic_layer.agent.planner import Intent, build_plan


def _stub_graph(monkeypatch, *, resolved=None, dim_targets=None, columns=None):
    """Stub the planner's graph reads so build_plan runs without Neo4j."""
    monkeypatch.setattr(planner_mod, "_resolve_values", lambda terms: resolved or [])
    monkeypatch.setattr(planner_mod, "_dimension_targets", lambda gb: dim_targets or [])
    monkeypatch.setattr(planner_mod, "_join_targets", lambda fact, ids: [
        {"table_id": t, "tables": [fact, t], "joins": []} for t in dict.fromkeys(ids)])
    monkeypatch.setattr(planner_mod, "_table_columns", lambda tid: columns or ["amount"])
    monkeypatch.setattr(planner_mod, "_context_docs", lambda terms: None)
    monkeypatch.setattr(planner_mod, "_api_correlations", lambda: [])


def test_build_plan_routing_disabled_is_unchanged(monkeypatch):
    _stub_graph(monkeypatch, dim_targets=["table:sales_pg.sales.segment"])
    monkeypatch.setattr(planner_mod.settings, "schema_routing_enabled", False, raising=False)
    called = {"routed": False}
    monkeypatch.setattr(planner_mod, "route_tables",
                        lambda *a, **k: called.__setitem__("routed", True) or [])
    plan = build_plan(Intent(group_by=["segment"], needs_sql=True))
    assert called["routed"] is False           # router not invoked when disabled
    assert plan["routed_tables"] == []
    targets = [jt["table_id"] for jt in plan["sql_legs"][0]["join_targets"]]
    assert "table:sales_pg.sales.segment" in targets


def test_build_plan_routing_enabled_unions_routed_tables(monkeypatch):
    _stub_graph(monkeypatch, dim_targets=["table:sales_pg.sales.segment"])
    monkeypatch.setattr(planner_mod.settings, "schema_routing_enabled", True, raising=False)
    monkeypatch.setattr(planner_mod, "route_tables",
                        lambda q, k_ret, k_rank: ["table:sales_pg.sales.region"])
    plan = build_plan(Intent(group_by=["segment"], needs_sql=True),
                      question="revenue by segment in EMEA")
    assert plan["routed_tables"] == ["table:sales_pg.sales.region"]
    targets = [jt["table_id"] for jt in plan["sql_legs"][0]["join_targets"]]
    assert "table:sales_pg.sales.region" in targets    # routed table folded into the join
    assert "table:sales_pg.sales.region" in plan["highlight"]


def test_build_plan_highlights_fact_table_without_dimensions(monkeypatch):
    _stub_graph(monkeypatch)
    monkeypatch.setattr(planner_mod.settings, "schema_routing_enabled", False, raising=False)

    plan = build_plan(Intent(needs_sql=True, fact="revenue"))

    assert "table:sales_pg.sales.order_line" in plan["highlight"]


def test_build_plan_highlights_api_correlation_endpoint_tables(monkeypatch):
    _stub_graph(monkeypatch)
    monkeypatch.setattr(planner_mod.settings, "schema_routing_enabled", False, raising=False)
    monkeypatch.setattr(planner_mod, "_api_correlations", lambda: [
        {
            "sql_column": "col:sales_pg.sales.customer.customer_id",
            "api_column": "col:itsm.api.GET /tickets.account_id",
        }
    ])

    plan = build_plan(Intent(needs_sql=False, needs_api=True, api_intents=["open tickets"]))

    assert "table:itsm.api.GET /tickets" in plan["highlight"]
