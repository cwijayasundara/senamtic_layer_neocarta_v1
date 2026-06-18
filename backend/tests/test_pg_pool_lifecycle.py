# backend/tests/test_pg_pool_lifecycle.py
from semantic_layer.agent import pg_pool


class _FakePool:
    def __init__(self):
        self.open_calls = 0
        self.close_calls = 0

    def open(self):
        self.open_calls += 1

    def close(self):
        self.close_calls += 1


def test_ensure_pool_open_opens_at_most_once(monkeypatch):
    fake = _FakePool()
    monkeypatch.setattr(pg_pool, "get_pool", lambda: fake)
    monkeypatch.setattr(pg_pool, "_pool_opened", False, raising=False)
    pg_pool.ensure_pool_open()
    pg_pool.ensure_pool_open()
    pg_pool.ensure_pool_open()
    assert fake.open_calls == 1   # guarded: opened once despite 3 calls


def test_app_lifespan_warms_and_closes_pool(monkeypatch):
    from fastapi.testclient import TestClient
    from semantic_layer.web import app as app_mod

    calls = {"open": 0, "close": 0}
    monkeypatch.setattr(app_mod, "ensure_pool_open", lambda: calls.__setitem__("open", calls["open"] + 1))

    class _P:
        def close(self):
            calls["close"] += 1

    monkeypatch.setattr(app_mod, "get_pool", lambda: _P())
    with TestClient(app_mod.app):          # context manager triggers lifespan startup/shutdown
        assert calls["open"] == 1
    assert calls["close"] == 1
