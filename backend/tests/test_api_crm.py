from fastapi.testclient import TestClient

from semantic_layer.apis.crm import crm_app

client = TestClient(crm_app)


def test_list_accounts_returns_all():
    r = client.get("/accounts")
    assert r.status_code == 200
    assert len(r.json()) == 40


def test_filter_accounts_by_region():
    r = client.get("/accounts", params={"region": "EMEA"})
    assert r.status_code == 200
    assert all(a["region"] == "EMEA" for a in r.json())


def test_get_account_by_id_and_404():
    assert client.get("/accounts/1").status_code == 200
    assert client.get("/accounts/99999").status_code == 404


def test_filter_opportunities_by_stage():
    r = client.get("/opportunities", params={"stage": "Negotiation"})
    assert r.status_code == 200
    assert all(o["stage"] == "Negotiation" for o in r.json())


def test_contacts_filter_by_account():
    r = client.get("/contacts", params={"account_id": 1})
    assert r.status_code == 200
    assert all(c["account_id"] == 1 for c in r.json())
