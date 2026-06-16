from fastapi.testclient import TestClient

from semantic_layer.apis.itsm import itsm_app

client = TestClient(itsm_app)


def test_list_tickets_returns_data():
    r = client.get("/tickets")
    assert r.status_code == 200
    assert len(r.json()) > 0


def test_filter_tickets_by_severity():
    r = client.get("/tickets", params={"severity": "Sev1"})
    assert r.status_code == 200
    assert all(t["severity"] == "Sev1" for t in r.json())


def test_filter_tickets_by_status_and_account():
    r = client.get("/tickets", params={"status": "Open", "account_id": 1})
    assert r.status_code == 200
    assert all(t["status"] == "Open" and t["account_id"] == 1 for t in r.json())


def test_get_ticket_404_for_missing():
    assert client.get("/tickets/99999").status_code == 404


def test_list_rmas():
    r = client.get("/rma")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
