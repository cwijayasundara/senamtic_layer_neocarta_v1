# backend/tests/test_api_federation.py
import json

from semantic_layer.config import settings
from semantic_layer.agent import api_tools


def test_api_source_list_parses_csv(monkeypatch):
    monkeypatch.setattr(settings, "api_sources", "crm, itsm ,newapi", raising=False)
    assert settings.api_source_list == ["crm", "itsm", "newapi"]


def test_call_api_validates_against_config(monkeypatch):
    monkeypatch.setattr(api_tools.settings, "api_sources", "crm,itsm,partner,dgx", raising=False)
    out = json.loads(api_tools.call_api.invoke({"source": "nope", "path": "/x"}))
    assert out["status"] == 404
    assert "unknown api source" in out["error"]
