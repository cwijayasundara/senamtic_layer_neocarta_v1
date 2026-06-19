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
