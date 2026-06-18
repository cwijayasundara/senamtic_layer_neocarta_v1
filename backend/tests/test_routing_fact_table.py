import pytest

from semantic_layer.agent import routing
from semantic_layer.agent import planner as planner_mod
from semantic_layer.agent.planner import Intent, build_plan


def test_select_fact_table_empty_returns_none():
    assert routing.select_fact_table([]) is None


def test_select_fact_table_no_sales_tables_returns_none():
    # Tables from other schemas must return None without touching the DB.
    result = routing.select_fact_table(
        ["table:other_db.x.y", "table:crm.api./accounts"]
    )
    assert result is None


def _stub_graph(monkeypatch, dim_targets):
    monkeypatch.setattr(planner_mod, "_resolve_values", lambda terms: [])
    monkeypatch.setattr(planner_mod, "_dimension_targets", lambda gb: dim_targets)
    monkeypatch.setattr(planner_mod, "_join_targets",
                        lambda fact, ids: [{"table_id": t, "tables": [fact, t], "joins": []}
                                           for t in dict.fromkeys(ids)])
    monkeypatch.setattr(planner_mod, "_table_columns", lambda tid: ["amount"])
    monkeypatch.setattr(planner_mod, "_context_docs", lambda terms: None)
    monkeypatch.setattr(planner_mod, "_api_correlations", lambda: [])


def test_build_plan_uses_selected_fact_table_when_routing_on(monkeypatch):
    _stub_graph(monkeypatch, dim_targets=["table:sales_pg.sales.segment"])
    monkeypatch.setattr(planner_mod.settings, "schema_routing_enabled", True, raising=False)
    monkeypatch.setattr(planner_mod, "route_tables",
                        lambda q, k_ret, k_rank: ["table:sales_pg.sales.invoice_line"])
    monkeypatch.setattr(planner_mod, "select_fact_table",
                        lambda routed: "table:sales_pg.sales.invoice_line")
    plan = build_plan(Intent(group_by=["segment"], needs_sql=True), question="revenue by segment")
    assert plan["sql_legs"][0]["fact_table"] == "table:sales_pg.sales.invoice_line"


def test_build_plan_keeps_default_fact_when_routing_off(monkeypatch):
    _stub_graph(monkeypatch, dim_targets=["table:sales_pg.sales.segment"])
    monkeypatch.setattr(planner_mod.settings, "schema_routing_enabled", False, raising=False)
    plan = build_plan(Intent(group_by=["segment"], needs_sql=True))
    assert plan["sql_legs"][0]["fact_table"] == planner_mod._SALES_FACT


@pytest.mark.neo4j
def test_select_fact_table_picks_order_line(ingested_graph):
    # order_line and customer BOTH have 2 direct FKs in this schema; depth2 breaks the tie.
    routed = ["table:sales_pg.sales.region", "table:sales_pg.sales.order_line",
              "table:sales_pg.sales.customer"]
    result = routing.select_fact_table(routed)
    assert result == "table:sales_pg.sales.order_line"
    # Explicitly confirm customer (equal direct FKs, alphabetically earlier) did not win.
    assert result != "table:sales_pg.sales.customer"
