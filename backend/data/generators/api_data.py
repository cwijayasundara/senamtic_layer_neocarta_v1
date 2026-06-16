"""Deterministic synthetic data for the four mock enterprise APIs.

Accounts reuse Plan 1's sales customer base (same ids and names), so API
records can be joined back to the relational `sales` sources by account
identity. Each generator is pure (no I/O) and deterministic for a given seed.
"""

import random
from datetime import date, timedelta

from data.generators.sales import generate_sales

OPP_STAGES = [
    "Prospecting",
    "Qualification",
    "Proposal",
    "Negotiation",
    "Closed Won",
    "Closed Lost",
]
ACCOUNT_TIERS = ["Strategic", "Enterprise", "Commercial"]
CONTACT_TITLES = [
    "CTO",
    "VP Infrastructure",
    "ML Platform Lead",
    "Procurement Manager",
    "Data Center Architect",
]
TICKET_SEVERITIES = ["Sev1", "Sev2", "Sev3", "Sev4"]
TICKET_STATUSES = ["Open", "In Progress", "Resolved", "Closed"]
SLA_BY_SEVERITY = {"Sev1": 4, "Sev2": 8, "Sev3": 24, "Sev4": 72}
RMA_STATUSES = ["Requested", "Approved", "Shipped", "Received", "Replaced"]
PARTNER_NAMES = [
    "Arrow Electronics",
    "TD SYNNEX",
    "Ingram Micro",
    "Insight Enterprises",
    "CDW",
]
PARTNER_TIERS = ["Elite", "Preferred", "Authorized"]
INSTANCE_TYPES = ["DGX H100", "DGX B200", "DGX A100", "DGX GB200"]


def build_accounts(seed=42):
    """Derive CRM accounts from the sales customer base. Returns (accounts, sales_data)."""
    data = generate_sales(seed=seed)
    region_by_country = {c["country_id"]: c["region_id"] for c in data["countries"]}
    region_name = {r["region_id"]: r["name"] for r in data["regions"]}
    industry_name = {i["industry_id"]: i["name"] for i in data["industries"]}
    rng = random.Random(seed + 1)
    accounts = [
        {
            "account_id": c["customer_id"],
            "name": c["name"],
            "industry": industry_name[c["industry_id"]],
            "region": region_name[region_by_country[c["country_id"]]],
            "tier": rng.choice(ACCOUNT_TIERS),
        }
        for c in data["customers"]
    ]
    return accounts, data


def generate_crm(seed=42):
    accounts, data = build_accounts(seed)
    rng = random.Random(seed + 2)
    product_lines = [pl["name"] for pl in data["product_lines"]]

    contacts = []
    cid = 0
    for a in accounts:
        for _ in range(rng.randint(1, 2)):
            cid += 1
            contacts.append(
                {
                    "contact_id": cid,
                    "account_id": a["account_id"],
                    "name": f"{a['name'].split()[0]} Contact {cid}",
                    "title": rng.choice(CONTACT_TITLES),
                    "email": f"contact{cid}@example.com",
                }
            )

    opportunities = []
    oid = 0
    for a in accounts:
        for _ in range(rng.randint(1, 3)):
            oid += 1
            opportunities.append(
                {
                    "opportunity_id": oid,
                    "account_id": a["account_id"],
                    "name": f"{rng.choice(product_lines)} expansion",
                    "stage": rng.choice(OPP_STAGES),
                    "amount": rng.randint(100, 5000) * 1000,
                    "product_line": rng.choice(product_lines),
                    "close_date": (date(2024, 1, 1) + timedelta(days=rng.randint(0, 540))).isoformat(),
                }
            )

    return {"accounts": accounts, "contacts": contacts, "opportunities": opportunities}


def generate_itsm(seed=42):
    accounts, data = build_accounts(seed)
    rng = random.Random(seed + 3)
    product_lines = [pl["name"] for pl in data["product_lines"]]

    tickets = []
    tid = 0
    for a in accounts:
        for _ in range(rng.randint(0, 4)):
            tid += 1
            sev = rng.choice(TICKET_SEVERITIES)
            tickets.append(
                {
                    "ticket_id": tid,
                    "account_id": a["account_id"],
                    "subject": f"{rng.choice(product_lines)} issue",
                    "severity": sev,
                    "status": rng.choice(TICKET_STATUSES),
                    "sla_hours": SLA_BY_SEVERITY[sev],
                    "product_line": rng.choice(product_lines),
                    "opened_at": (date(2024, 6, 1) + timedelta(days=rng.randint(0, 365))).isoformat(),
                }
            )

    rmas = []
    rid = 0
    for t in tickets:
        if rng.random() < 0.2:
            rid += 1
            rmas.append(
                {
                    "rma_id": rid,
                    "ticket_id": t["ticket_id"],
                    "product": t["product_line"],
                    "serial": f"SN-{rng.randint(100000, 999999)}",
                    "status": rng.choice(RMA_STATUSES),
                }
            )

    return {"tickets": tickets, "rmas": rmas}


def generate_partner(seed=42):
    accounts, data = build_accounts(seed)
    rng = random.Random(seed + 4)
    regions = [r["name"] for r in data["regions"]]
    product_lines = [pl["name"] for pl in data["product_lines"]]

    partners = [
        {
            "partner_id": i,
            "name": name,
            "region": rng.choice(regions),
            "tier": rng.choice(PARTNER_TIERS),
        }
        for i, name in enumerate(PARTNER_NAMES, start=1)
    ]

    inventory = []
    iid = 0
    for p in partners:
        for pl in product_lines:
            iid += 1
            on_hand = rng.randint(0, 500)
            allocated = rng.randint(0, on_hand)
            inventory.append(
                {
                    "inventory_id": iid,
                    "partner_id": p["partner_id"],
                    "product_line": pl,
                    "on_hand": on_hand,
                    "allocated": allocated,
                    "available": on_hand - allocated,
                }
            )

    return {"partners": partners, "inventory": inventory}


def generate_dgx(seed=42):
    accounts, _ = build_accounts(seed)
    rng = random.Random(seed + 5)
    start = date(2025, 1, 1)

    usage = []
    uid = 0
    for a in accounts:
        if rng.random() < 0.5:  # roughly half the accounts use DGX Cloud
            continue
        day = 0
        while day < 90:
            uid += 1
            usage.append(
                {
                    "usage_id": uid,
                    "account_id": a["account_id"],
                    "instance_type": rng.choice(INSTANCE_TYPES),
                    "gpu_hours": round(rng.uniform(10, 2000), 1),
                    "utilization_pct": round(rng.uniform(20, 99), 1),
                    "usage_date": (start + timedelta(days=day)).isoformat(),
                }
            )
            day += rng.randint(5, 10)

    return {"usage": usage}
