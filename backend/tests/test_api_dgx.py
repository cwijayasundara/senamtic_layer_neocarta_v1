from fastapi.testclient import TestClient

from semantic_layer.apis.dgx import dgx_app

client = TestClient(dgx_app)


def test_list_usage_returns_data():
    r = client.get("/usage")
    assert r.status_code == 200
    assert len(r.json()) > 0


def test_filter_usage_by_instance_type():
    r = client.get("/usage", params={"instance_type": "DGX B200"})
    assert r.status_code == 200
    assert all(u["instance_type"] == "DGX B200" for u in r.json())


def test_filter_usage_by_account_and_date_range():
    r = client.get(
        "/usage",
        params={"account_id": 1, "start": "2025-01-01", "end": "2025-02-01"},
    )
    assert r.status_code == 200
    for u in r.json():
        assert u["account_id"] == 1
        assert "2025-01-01" <= u["usage_date"] <= "2025-02-01"
