from fastapi.testclient import TestClient

from semantic_layer.apis.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_mounted_subapps_serve_data():
    assert client.get("/crm/accounts").status_code == 200
    assert client.get("/itsm/tickets").status_code == 200
    assert client.get("/partner/partners").status_code == 200
    assert client.get("/dgx/usage").status_code == 200


def test_each_subapp_exposes_its_own_openapi_spec():
    specs = {
        "/crm/openapi.json": "/accounts",
        "/itsm/openapi.json": "/tickets",
        "/partner/openapi.json": "/partners",
        "/dgx/openapi.json": "/usage",
    }
    for spec_url, expected_path in specs.items():
        r = client.get(spec_url)
        assert r.status_code == 200
        body = r.json()
        assert "openapi" in body
        assert expected_path in body["paths"]
