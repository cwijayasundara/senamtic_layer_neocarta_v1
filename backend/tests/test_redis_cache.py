# backend/tests/test_redis_cache.py
import fakeredis

from semantic_layer.agent.redis_cache import RedisQueryCache


def _cache():
    return RedisQueryCache(fakeredis.FakeStrictRedis(), ttl_seconds=1000)


def test_put_then_get_exact_roundtrip_normalized():
    c = _cache()
    c.put("What is revenue?", [{"type": "answer", "content": "42"}])
    assert c.get_exact("  what   IS revenue? ") == [{"type": "answer", "content": "42"}]


def test_get_exact_miss_returns_none():
    assert _cache().get_exact("nothing here") is None


def test_get_semantic_is_none_in_redis_backend():
    c = _cache()
    c.put("q", [{"type": "answer", "content": "x"}], embedding=[0.1, 0.2])
    assert c.get_semantic([0.1, 0.2], 0.9) is None   # cross-worker semantic deferred


def test_put_sets_ttl():
    client = fakeredis.FakeStrictRedis()
    RedisQueryCache(client, ttl_seconds=123).put("q", [{"type": "answer", "content": "x"}])
    # one key written, with a TTL close to the configured ttl
    keys = client.keys("qcache:*")
    assert len(keys) == 1
    assert 0 < client.ttl(keys[0]) <= 123


# --- Factory tests (Task A2) ---
import fakeredis as _fr

from semantic_layer.agent import cache as cache_mod
from semantic_layer.agent.cache import build_query_cache, QueryCache
from semantic_layer.agent.redis_cache import RedisQueryCache as _RQC


def test_factory_defaults_to_memory(monkeypatch):
    monkeypatch.setattr(cache_mod.settings, "cache_backend", "memory", raising=False)
    assert isinstance(build_query_cache(), QueryCache)


def test_factory_builds_redis_when_configured(monkeypatch):
    monkeypatch.setattr(cache_mod.settings, "cache_backend", "redis", raising=False)
    monkeypatch.setattr(cache_mod, "_redis_client_from_url", lambda url: _fr.FakeStrictRedis())
    c = build_query_cache()
    assert isinstance(c, _RQC)
    c.put("q", [{"type": "answer", "content": "ok"}])
    assert c.get_exact("q") == [{"type": "answer", "content": "ok"}]


import pytest


@pytest.mark.redis
def test_real_redis_roundtrip():
    import redis as _redis
    from semantic_layer.config import settings
    try:
        client = _redis.from_url(settings.redis_url)
        client.ping()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Redis not available: {exc}")
    c = _RQC(client, ttl_seconds=60)
    c.put("integration q", [{"type": "answer", "content": "real"}])
    assert c.get_exact("integration q") == [{"type": "answer", "content": "real"}]
    client.delete("qcache:integration q")
