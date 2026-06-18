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


@pytest.mark.neo4j
def test_select_fact_table_depth2_excludes_self_reference(ingested_graph):
    # Build two sales-schema probe tables that BOTH have 1 direct FK.
    # selfy: FK to mid1; mid1 has a FK back to selfy (circular).
    #   Without the t2<>t guard selfy has depth2=1 (reaches itself) and wins over reachy.
    #   With the guard the self-loop is excluded, selfy.depth2=0, and reachy wins.
    # reachy: FK to mid2 (isolated; mid2 has no outgoing FKs) => depth2=0 in both cases.
    with ingested_graph.session() as s:
        s.run(
            """
            MERGE (selfy:Table {id:'table:sales_pg.sales._probe_selfy'})
            MERGE (reachy:Table {id:'table:sales_pg.sales._probe_reachy'})
            MERGE (mid1:Table {id:'table:sales_pg.sales._probe_mid1'})
            MERGE (mid2:Table {id:'table:sales_pg.sales._probe_mid2'})
            // selfy: 1 FK to mid1
            MERGE (selfy)-[:HAS_COLUMN]->(cs1:Column {id:'col:_p.selfy.fk_mid1'})
            MERGE (mid1)-[:HAS_COLUMN]->(cm1pk:Column {id:'col:_p.mid1.pk'})
            MERGE (cs1)-[:REFERENCES]->(cm1pk)
            // selfy has a PK column
            MERGE (selfy)-[:HAS_COLUMN]->(csp:Column {id:'col:_p.selfy.pk'})
            // mid1 has a FK back to selfy (creates circular depth-2 path for selfy)
            MERGE (mid1)-[:HAS_COLUMN]->(cm1fk:Column {id:'col:_p.mid1.fk_selfy'})
            MERGE (cm1fk)-[:REFERENCES]->(csp)
            // reachy: 1 FK to mid2 (mid2 has no outgoing FKs => reachy.depth2=0)
            MERGE (reachy)-[:HAS_COLUMN]->(cr1:Column {id:'col:_p.reachy.fk_mid2'})
            MERGE (mid2)-[:HAS_COLUMN]->(cm2pk:Column {id:'col:_p.mid2.pk'})
            MERGE (cr1)-[:REFERENCES]->(cm2pk)
            """
        )
    try:
        result = routing.select_fact_table([
            "table:sales_pg.sales._probe_selfy", "table:sales_pg.sales._probe_reachy"])
        # Without guard: selfy.depth2=1 (self-loop via mid1) > reachy.depth2=0 => selfy wins.
        # With guard (t2<>t): selfy.depth2=0 = reachy.depth2=0, tie broken by tid ASC => reachy wins.
        assert result == "table:sales_pg.sales._probe_reachy"
    finally:
        with ingested_graph.session() as s:
            s.run(
                "MATCH (t:Table) WHERE t.id STARTS WITH 'table:sales_pg.sales._probe_' "
                "OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c) DETACH DELETE t, c")
