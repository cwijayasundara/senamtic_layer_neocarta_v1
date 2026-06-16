from semantic_layer.apis.app import app
from semantic_layer.ingest.api_extractor import extract_openapi, extract_all_apis


def _spec(prefix: str) -> dict:
    from fastapi.testclient import TestClient
    return TestClient(app).get(f"{prefix}/openapi.json").json()


def test_extract_crm_openapi_to_virtual_tables():
    bundle = extract_openapi(_spec("/crm"), source="crm")
    table_names = {t.name for t in bundle.tables}
    assert "GET /accounts" in table_names
    col_names = {c.name for c in bundle.columns}
    assert {"account_id", "name", "industry", "region"} <= col_names


def test_extract_all_apis_covers_four_sources():
    bundles = extract_all_apis(_spec, ("crm", "itsm", "partner", "dgx"))
    sources = {db.name for b in bundles for db in b.databases}
    assert sources == {"crm", "itsm", "partner", "dgx"}
