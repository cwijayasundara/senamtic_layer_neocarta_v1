from data.generators.api_data import (
    generate_crm,
    generate_itsm,
    generate_partner,
    generate_dgx,
)


def test_crm_is_deterministic_and_has_accounts():
    a = generate_crm(seed=42)
    b = generate_crm(seed=42)
    assert a == b
    assert len(a["accounts"]) == 40  # reuses the 40 sales customers


def test_crm_referential_integrity():
    crm = generate_crm(seed=42)
    account_ids = {a["account_id"] for a in crm["accounts"]}
    assert all(c["account_id"] in account_ids for c in crm["contacts"])
    assert all(o["account_id"] in account_ids for o in crm["opportunities"])


def test_itsm_tickets_reference_accounts_and_rmas_reference_tickets():
    crm = generate_crm(seed=42)
    itsm = generate_itsm(seed=42)
    account_ids = {a["account_id"] for a in crm["accounts"]}
    ticket_ids = {t["ticket_id"] for t in itsm["tickets"]}
    assert all(t["account_id"] in account_ids for t in itsm["tickets"])
    assert all(r["ticket_id"] in ticket_ids for r in itsm["rmas"])
    assert all(t["severity"] in {"Sev1", "Sev2", "Sev3", "Sev4"} for t in itsm["tickets"])


def test_partner_inventory_available_is_consistent():
    partner = generate_partner(seed=42)
    partner_ids = {p["partner_id"] for p in partner["partners"]}
    assert len(partner["partners"]) == 5
    for inv in partner["inventory"]:
        assert inv["partner_id"] in partner_ids
        assert inv["available"] == inv["on_hand"] - inv["allocated"]
        assert inv["available"] >= 0


def test_dgx_usage_present_and_valid():
    crm = generate_crm(seed=42)
    dgx = generate_dgx(seed=42)
    account_ids = {a["account_id"] for a in crm["accounts"]}
    assert len(dgx["usage"]) > 0
    for u in dgx["usage"]:
        assert u["account_id"] in account_ids
        assert u["gpu_hours"] > 0
        assert 0 <= u["utilization_pct"] <= 100
        assert u["instance_type"] in {"DGX H100", "DGX B200", "DGX A100", "DGX GB200"}


def test_all_generators_deterministic():
    assert generate_itsm(seed=42) == generate_itsm(seed=42)
    assert generate_partner(seed=42) == generate_partner(seed=42)
    assert generate_dgx(seed=42) == generate_dgx(seed=42)
