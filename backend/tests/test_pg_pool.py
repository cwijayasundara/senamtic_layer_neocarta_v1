from semantic_layer.agent import pg_pool


def test_get_pool_is_cached_singleton():
    pg_pool.get_pool.cache_clear()
    p1 = pg_pool.get_pool()
    p2 = pg_pool.get_pool()
    assert p1 is p2                      # cached: one pool per process


def test_get_pool_uses_configured_sizes():
    from semantic_layer.config import settings
    pg_pool.get_pool.cache_clear()
    p = pg_pool.get_pool()
    assert p.max_size == settings.pg_pool_max_size
    assert p.min_size == settings.pg_pool_min_size
