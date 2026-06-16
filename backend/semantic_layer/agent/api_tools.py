"""Tool to call the mock enterprise REST APIs (in-process via TestClient)."""

import json

from fastapi.testclient import TestClient
from langchain_core.tools import tool

from semantic_layer.apis.app import app

_client = TestClient(app)
_SOURCES = {"crm", "itsm", "partner", "dgx"}


@tool
def call_api(source: str, path: str, params: dict | None = None) -> str:
    """Call a mock enterprise API and return its JSON.

    source is one of crm, itsm, partner, dgx. path is the endpoint under that API
    (e.g. '/tickets', '/accounts', '/inventory', '/usage'). params is an optional
    dict of query filters. Returns {status, data}. Use get_table_schema / the API's
    virtual tables to learn the available endpoints and fields."""
    if source not in _SOURCES:
        return json.dumps({"status": 404, "error": f"unknown api source '{source}'"})
    resp = _client.get(f"/{source}{path}", params=params or {})
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001
        body = resp.text
    return json.dumps({"status": resp.status_code, "data": body})
