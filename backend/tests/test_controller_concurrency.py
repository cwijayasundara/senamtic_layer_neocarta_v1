# backend/tests/test_controller_concurrency.py
from semantic_layer.agent import controller as ctrl
from semantic_layer.agent import cache as cache_mod
from semantic_layer.agent.planner import Intent


class _CountingGate:
    def __init__(self):
        self.acquired = 0
        self.released = 0

    def __enter__(self):
        self.acquired += 1
        return self

    def __exit__(self, *exc):
        self.released += 1
        return False


def _stub_live_run(monkeypatch):
    monkeypatch.setattr(ctrl, "extract_intent", lambda q: Intent(needs_sql=False))
    monkeypatch.setattr(ctrl, "build_plan", lambda intent, question=None: {
        "highlight": [], "sql_legs": [], "doc_leg": None, "api_correlations": []})
    monkeypatch.setattr(ctrl, "_synthesize", lambda *a, **k: "answer")
    monkeypatch.setattr(ctrl, "check_numeric_grounding", lambda *a, **k: [])


def test_live_run_acquires_and_releases_gate_once(monkeypatch):
    monkeypatch.setattr(ctrl, "query_cache", cache_mod.QueryCache(max_entries=10, ttl_seconds=1000))
    monkeypatch.setattr(ctrl.settings, "query_cache_enabled", False, raising=False)
    gate = _CountingGate()
    monkeypatch.setattr(ctrl, "_answer_gate", gate)
    _stub_live_run(monkeypatch)
    list(ctrl.answer_stream("anything"))
    assert gate.acquired == 1
    assert gate.released == 1


def test_cache_hit_does_not_acquire_gate(monkeypatch):
    fresh = cache_mod.QueryCache(max_entries=10, ttl_seconds=1000)
    monkeypatch.setattr(ctrl, "query_cache", fresh)
    monkeypatch.setattr(ctrl.settings, "query_cache_enabled", True, raising=False)
    gate = _CountingGate()
    monkeypatch.setattr(ctrl, "_answer_gate", gate)
    _stub_live_run(monkeypatch)
    list(ctrl.answer_stream("q"))        # populate (acquires once)
    assert gate.acquired == 1
    list(ctrl.answer_stream("q"))        # exact hit: must NOT acquire again
    assert gate.acquired == 1
