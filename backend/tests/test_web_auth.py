import pytest
from fastapi import HTTPException

from semantic_layer.web import auth as auth_mod


def test_require_api_key_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(auth_mod.settings, "api_keys", "", raising=False)
    assert auth_mod.require_api_key(x_api_key=None) is None   # disabled -> allow


def test_require_api_key_rejects_missing_and_bad(monkeypatch):
    monkeypatch.setattr(auth_mod.settings, "api_keys", "good-key,other", raising=False)
    with pytest.raises(HTTPException) as e1:
        auth_mod.require_api_key(x_api_key=None)
    assert e1.value.status_code == 401
    with pytest.raises(HTTPException) as e2:
        auth_mod.require_api_key(x_api_key="nope")
    assert e2.value.status_code == 401


def test_require_api_key_accepts_listed_key(monkeypatch):
    monkeypatch.setattr(auth_mod.settings, "api_keys", "good-key", raising=False)
    assert auth_mod.require_api_key(x_api_key="good-key") is None


def test_graph_endpoint_enforces_key(monkeypatch):
    from fastapi.testclient import TestClient
    from semantic_layer.web import app as app_mod
    monkeypatch.setattr(app_mod.settings, "api_keys", "secret", raising=False)
    monkeypatch.setattr(app_mod, "get_schema_graph",
                        lambda source=None, max_chunks=None: {"nodes": [], "edges": [], "truncated": False})
    client = TestClient(app_mod.app)
    assert client.get("/graph").status_code == 401
    assert client.get("/graph", headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/health").status_code == 200   # health stays open
