# backend/tests/test_api_federation.py
import json

import pytest

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


@pytest.mark.neo4j
def test_route_api_endpoints_finds_ticket_endpoint(ingested_graph):
    from semantic_layer.agent import routing
    eps = routing.route_api_endpoints(["open tickets"])
    # ITSM's /tickets endpoint should surface for a ticket-related intent.
    assert any(e["source"] == "itsm" and e["path"] == "/tickets" for e in eps)
    for e in eps:
        assert set(e) >= {"source", "path", "summary"}


def test_run_api_leg_uses_routed_endpoints(monkeypatch):
    from semantic_layer.agent import legs as legs_mod
    monkeypatch.setattr(legs_mod, "route_api_endpoints",
                        lambda intents, limit=12: [{"source": "itsm", "path": "/tickets",
                                                    "summary": "List support tickets"}])
    plan_calls = legs_mod._ApiCalls(calls=[
        legs_mod._ApiCall(source="itsm", path="/tickets", params={"status": "open"})])
    captured = {}

    class _FakeStructured:
        def invoke(self, messages):
            captured["human"] = messages[-1][1]
            return plan_calls

    class _FakeModel:
        def with_structured_output(self, _schema, **_kw):
            return _FakeStructured()

    monkeypatch.setattr(legs_mod, "get_chat_model", lambda model=None: _FakeModel())
    monkeypatch.setattr(legs_mod, "call_api", type("T", (), {
        "invoke": staticmethod(lambda _a: json.dumps({"status": 200, "data": [{"id": 1}]}))})())
    out = legs_mod.run_api_leg(["open tickets"])
    assert out["error"] is None
    assert out["calls"][0]["path"] == "/tickets"
    # the routed endpoint was injected into the prompt the model saw
    assert "/tickets" in captured["human"]


def test_api_leg_federates_past_demo_apis_and_drops_static_list(monkeypatch):
    """A routed endpoint from a NON-default API reaches the model, and the system
    prompt no longer pins the four demo APIs (the routed catalog is authoritative)."""
    from semantic_layer.agent import legs as legs_mod

    # The static enumeration must be gone from the system prompt.
    assert "/accounts" not in legs_mod._API_LEG_PROMPT
    assert "dgx (/usage)" not in legs_mod._API_LEG_PROMPT

    monkeypatch.setattr(legs_mod, "route_api_endpoints",
                        lambda intents, limit=12: [{"source": "billing", "path": "/invoices",
                                                    "summary": "List customer invoices"}])
    plan_calls = legs_mod._ApiCalls(calls=[
        legs_mod._ApiCall(source="billing", path="/invoices", params={})])
    captured = {}

    class _FakeStructured:
        def invoke(self, messages):
            captured["system"] = messages[0][1]
            captured["human"] = messages[-1][1]
            return plan_calls

    class _FakeModel:
        def with_structured_output(self, _schema, **_kw):
            return _FakeStructured()

    monkeypatch.setattr(legs_mod, "get_chat_model", lambda model=None: _FakeModel())
    monkeypatch.setattr(legs_mod, "call_api", type("T", (), {
        "invoke": staticmethod(lambda _a: __import__("json").dumps({"status": 200, "data": [{"id": 1}]}))})())
    out = legs_mod.run_api_leg(["customer invoices"])
    assert out["error"] is None
    # the federated (non-demo) endpoint reached the model via the routed catalog
    assert "billing" in captured["human"] and "/invoices" in captured["human"]
    assert out["calls"][0]["source"] == "billing"
