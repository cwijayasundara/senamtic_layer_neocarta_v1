import json

from semantic_layer.agent.api_tools import call_api


def test_call_api_lists_tickets():
    out = json.loads(call_api.invoke({"source": "itsm", "path": "/tickets", "params": {"severity": "Sev1"}}))
    assert out["status"] == 200
    assert isinstance(out["data"], list)
    assert all(t["severity"] == "Sev1" for t in out["data"])


def test_call_api_unknown_source():
    out = json.loads(call_api.invoke({"source": "nope", "path": "/x", "params": {}}))
    assert out["status"] >= 400
