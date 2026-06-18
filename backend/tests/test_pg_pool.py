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


import json

import pytest


@pytest.mark.postgres
def test_run_sales_pg_uses_pool_and_returns_rows(postgres_dsn, monkeypatch):
    from semantic_layer.agent import sql_tools
    from semantic_layer.agent import pg_pool

    pg_pool.get_pool.cache_clear()
    used = {"pool": False}
    real_get_pool = pg_pool.get_pool

    def tracking_get_pool():
        used["pool"] = True
        return real_get_pool()

    monkeypatch.setattr(sql_tools, "get_pool", tracking_get_pool)
    out = json.loads(sql_tools._run("sales_pg", "SELECT 1 AS n"))
    assert used["pool"] is True
    assert out["columns"] == ["n"]
    assert out["rows"] == [[1]]
