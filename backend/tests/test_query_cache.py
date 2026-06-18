# backend/tests/test_query_cache.py
from semantic_layer.agent.cache import QueryCache


def test_exact_hit_after_put():
    c = QueryCache(max_entries=10, ttl_seconds=1000)
    c.put("What is revenue?", {"content": "42"})
    assert c.get_exact("  what   IS revenue? ") == {"content": "42"}   # normalized match


def test_exact_miss_returns_none():
    c = QueryCache(max_entries=10, ttl_seconds=1000)
    assert c.get_exact("anything") is None


def test_lru_eviction():
    c = QueryCache(max_entries=2, ttl_seconds=1000)
    c.put("a", {"content": "A"})
    c.put("b", {"content": "B"})
    c.put("c", {"content": "C"})            # evicts "a" (oldest)
    assert c.get_exact("a") is None
    assert c.get_exact("b") == {"content": "B"}
    assert c.get_exact("c") == {"content": "C"}


def test_ttl_expiry():
    clock = {"t": 1000.0}
    c = QueryCache(max_entries=10, ttl_seconds=5, now=lambda: clock["t"])
    c.put("q", {"content": "X"})
    clock["t"] = 1004.0
    assert c.get_exact("q") == {"content": "X"}   # within TTL
    clock["t"] = 1006.0
    assert c.get_exact("q") is None               # expired


from semantic_layer.agent import controller as ctrl
from semantic_layer.agent import cache as cache_mod
from semantic_layer.agent.planner import Intent


def test_answer_stream_serves_exact_cache_hit(monkeypatch):
    fresh = cache_mod.QueryCache(max_entries=10, ttl_seconds=1000)
    monkeypatch.setattr(ctrl, "query_cache", fresh)
    monkeypatch.setattr(ctrl.settings, "query_cache_enabled", True, raising=False)
    calls = {"intent": 0}
    monkeypatch.setattr(ctrl, "extract_intent",
                        lambda q: calls.__setitem__("intent", calls["intent"] + 1) or Intent())
    monkeypatch.setattr(ctrl, "build_plan", lambda intent, question=None: {
        "highlight": [], "sql_legs": [], "doc_leg": None, "api_correlations": []})
    monkeypatch.setattr(ctrl, "_synthesize", lambda *a, **k: "cached-me")
    monkeypatch.setattr(ctrl, "check_numeric_grounding", lambda *a, **k: [])

    first = list(ctrl.answer_stream("Total revenue?"))
    assert first[-1]["content"] == "cached-me"
    assert calls["intent"] == 1

    second = list(ctrl.answer_stream("  total   revenue? "))   # normalized same question
    assert second[-1]["type"] == "answer"
    assert second[-1]["content"] == "cached-me"
    assert calls["intent"] == 1                                # legs NOT re-run on hit
