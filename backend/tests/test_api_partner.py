from fastapi.testclient import TestClient

from semantic_layer.apis.partner import partner_app

client = TestClient(partner_app)


def test_list_partners():
    r = client.get("/partners")
    assert r.status_code == 200
    assert len(r.json()) == 5


def test_list_inventory_and_available_consistency():
    r = client.get("/inventory")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) > 0
    assert all(i["available"] == i["on_hand"] - i["allocated"] for i in rows)


def test_filter_inventory_by_partner_and_product_line():
    r = client.get("/inventory", params={"partner_id": 1, "product_line": "B200"})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) > 0
    assert all(i["partner_id"] == 1 and i["product_line"] == "B200" for i in rows)
