# backend/tests/test_routing_activation.py
from semantic_layer.agent import planner as planner_mod
from semantic_layer.agent.planner import Intent, build_plan


def _stub_graph(monkeypatch, dim_targets):
    monkeypatch.setattr(planner_mod, "_resolve_values", lambda terms: [])
    monkeypatch.setattr(planner_mod, "_dimension_targets", lambda gb: dim_targets)
    monkeypatch.setattr(planner_mod, "_join_targets",
                        lambda fact, ids: [{"table_id": t, "tables": [fact, t], "joins": []}
                                           for t in dict.fromkeys(ids)])
    monkeypatch.setattr(planner_mod, "_table_columns", lambda tid: ["amount"])
    monkeypatch.setattr(planner_mod, "_context_docs", lambda terms: None)
    monkeypatch.setattr(planner_mod, "_api_correlations", lambda: [])
    monkeypatch.setattr(planner_mod, "select_fact_table", lambda routed: None)


def test_routing_on_bounds_targets_to_routed_and_cap(monkeypatch):
    # dimension scan finds 3 sales tables; routing returns only 1 of them -> bounded to that 1.
    _stub_graph(monkeypatch, dim_targets=[
        "table:sales_pg.sales.segment", "table:sales_pg.sales.region",
        "table:sales_pg.sales.industry"])
    monkeypatch.setattr(planner_mod.settings, "schema_routing_enabled", True, raising=False)
    monkeypatch.setattr(planner_mod.settings, "schema_routing_max_targets", 8, raising=False)
    monkeypatch.setattr(planner_mod, "route_tables",
                        lambda q, k_ret, k_rank: ["table:sales_pg.sales.segment"])
    plan = build_plan(Intent(group_by=["segment"], needs_sql=True), question="rev by segment")
    targets = [jt["table_id"] for jt in plan["sql_legs"][0]["join_targets"]]
    assert targets == ["table:sales_pg.sales.segment"]   # bounded to the routed table


def test_routing_off_is_unchanged(monkeypatch):
    _stub_graph(monkeypatch, dim_targets=["table:sales_pg.sales.segment",
                                          "table:sales_pg.sales.region"])
    monkeypatch.setattr(planner_mod.settings, "schema_routing_enabled", False, raising=False)
    plan = build_plan(Intent(group_by=["segment", "region"], needs_sql=True))
    targets = {jt["table_id"] for jt in plan["sql_legs"][0]["join_targets"]}
    assert targets == {"table:sales_pg.sales.segment", "table:sales_pg.sales.region"}


from semantic_layer.eval import compare as compare_mod
from semantic_layer.eval.compare import compare_routing


def test_compare_routing_runs_both_modes_and_restores(monkeypatch):
    monkeypatch.setattr(compare_mod.settings, "schema_routing_enabled", False, raising=False)
    seen = []

    def fake_run(evalset):
        seen.append(compare_mod.settings.schema_routing_enabled)
        # mean depends on the flag so we can tell the modes apart
        return {"results": [], "mean_score": 4.0 if compare_mod.settings.schema_routing_enabled else 3.0,
                "pass_rate": 1.0}

    out = compare_routing([{"id": "x", "question": "q", "expect": "e"}], run_fn=fake_run)
    assert seen == [False, True]                       # ran OFF then ON
    assert out["off"]["mean_score"] == 3.0
    assert out["on"]["mean_score"] == 4.0
    assert out["delta_mean"] == 1.0                    # on - off
    assert compare_mod.settings.schema_routing_enabled is False   # restored
