# Plan 2: Mock Enterprise APIs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up four NVIDIA-themed mock enterprise REST APIs (CRM, Support/ITSM, Partner inventory, DGX Cloud telemetry) as mounted FastAPI sub-applications served on a single port, each with its own OpenAPI spec and deterministic synthetic data, so that later plans can (a) introspect each API's OpenAPI spec into the semantic graph and (b) let the agent answer questions by calling these APIs.

**Architecture:** A pure-Python data layer (`data/generators/api_data.py`) generates deterministic records for all four APIs, reusing Plan 1's sales customer base so API accounts share identity with the relational `sales.customer` rows (enabling cross-source joins by account). Pydantic response models live in `semantic_layer/apis/models.py`; an `lru_cache`-backed `store.py` builds the data once. Each API is its own `FastAPI()` instance (`crm.py`, `itsm.py`, `partner.py`, `dgx.py`) mounted under a path prefix by `app.py`, so every API exposes an independent `/{prefix}/openapi.json`. Tests use `fastapi.testclient.TestClient` (no database needed).

**Tech Stack:** Python 3.11, FastAPI, Uvicorn, Pydantic v2, `httpx` (for TestClient), `pytest`. Builds on Plan 1's `backend/` package, config, and data generators.

**Prerequisite:** Plan 1 (Data Foundation) is merged/present. This plan adds only new files plus small edits to `pyproject.toml`, `Makefile`, and `backend/README.md`. No Postgres/Neo4j required to run or test Plan 2 (the APIs serve in-memory synthetic data).

This is sub-plan 2 of 5 (Data Foundation → **Mock APIs** → Graph Ingestion → Agent → Web App).

---

## File Structure

```
backend/
  pyproject.toml                         # (modify) add fastapi, uvicorn, httpx
  data/generators/
    api_data.py                          # (new) deterministic data for all 4 APIs, reusing sales customers
  semantic_layer/apis/
    __init__.py                          # (new) empty
    models.py                            # (new) pydantic response models for all entities
    store.py                             # (new) lru_cache-backed builders calling api_data generators
    crm.py                               # (new) CRM FastAPI sub-app
    itsm.py                              # (new) Support/ITSM FastAPI sub-app
    partner.py                           # (new) Partner inventory FastAPI sub-app
    dgx.py                               # (new) DGX Cloud telemetry FastAPI sub-app
    app.py                               # (new) main app mounting the 4 sub-apps + /health
  tests/
    test_api_data.py                     # (new) generator integrity + determinism
    test_api_crm.py                      # (new) TestClient tests for CRM
    test_api_itsm.py                     # (new) TestClient tests for ITSM
    test_api_partner.py                  # (new) TestClient tests for partner
    test_api_dgx.py                      # (new) TestClient tests for DGX
    test_api_app.py                      # (new) mounting + per-sub-app OpenAPI specs
Makefile                                 # (modify) add serve-apis target
backend/README.md                        # (modify) document the APIs + make serve-apis
```

**Responsibilities:** `api_data.py` is the only data-generation unit (pure, no I/O, no web). `models.py` defines the wire contract. `store.py` is the single place data is materialized. Each sub-app file owns exactly one API's routes. `app.py` only composes.

---

## Task 1: Add web dependencies and the API data generators

**Files:**
- Modify: `backend/pyproject.toml`
- Create: `backend/data/generators/api_data.py`
- Test: `backend/tests/test_api_data.py`

- [ ] **Step 1: Add dependencies to `backend/pyproject.toml`**

In the `[project]` `dependencies` list add `fastapi>=0.110` and `uvicorn[standard]>=0.29`; in `[project.optional-dependencies].dev` add `httpx>=0.27`. The resulting two sections must read exactly:

```toml
dependencies = [
    "pydantic-settings>=2.2",
    "psycopg[binary]>=3.1",
    "Faker>=25.0",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "httpx>=0.27"]
```

- [ ] **Step 2: Install the new deps into the venv**

Run: `cd backend && ./.venv/bin/python -m pip install -e ".[dev]"`
Expected: fastapi, uvicorn, httpx install successfully.

- [ ] **Step 3: Write the failing test** `backend/tests/test_api_data.py`

```python
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
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd backend && ./.venv/bin/python -m pytest tests/test_api_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.generators.api_data'`

- [ ] **Step 5: Implement `backend/data/generators/api_data.py`**

```python
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
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd backend && ./.venv/bin/python -m pytest tests/test_api_data.py -v`
Expected: PASS (6 passed)

- [ ] **Step 7: Commit**

```bash
git add backend/pyproject.toml backend/data/generators/api_data.py backend/tests/test_api_data.py
git commit -m "feat(apis): deterministic data generators for the four mock enterprise APIs"
```

---

## Task 2: Pydantic response models and the data store

**Files:**
- Create: `backend/semantic_layer/apis/__init__.py`
- Create: `backend/semantic_layer/apis/models.py`
- Create: `backend/semantic_layer/apis/store.py`
- Test: `backend/tests/test_api_store.py`

- [ ] **Step 1: Create `backend/semantic_layer/apis/__init__.py`** (empty file)

```python
```

- [ ] **Step 2: Write the failing test** `backend/tests/test_api_store.py`

```python
from semantic_layer.apis import store
from semantic_layer.apis.models import Account, Ticket, InventoryItem, UsageRecord


def test_store_returns_cached_identical_objects():
    assert store.crm_data() is store.crm_data()  # lru_cache returns same object


def test_models_validate_store_rows():
    Account(**store.crm_data()["accounts"][0])
    Ticket(**store.itsm_data()["tickets"][0])
    InventoryItem(**store.partner_data()["inventory"][0])
    UsageRecord(**store.dgx_data()["usage"][0])
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd backend && ./.venv/bin/python -m pytest tests/test_api_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'semantic_layer.apis.store'`

- [ ] **Step 4: Implement `backend/semantic_layer/apis/models.py`**

```python
"""Pydantic response models for the mock enterprise APIs.

These drive each sub-app's OpenAPI schema, which Plan 3's NeoCarta API
extractor introspects to build virtual-table/column metadata nodes.
"""

from pydantic import BaseModel


class Account(BaseModel):
    account_id: int
    name: str
    industry: str
    region: str
    tier: str


class Contact(BaseModel):
    contact_id: int
    account_id: int
    name: str
    title: str
    email: str


class Opportunity(BaseModel):
    opportunity_id: int
    account_id: int
    name: str
    stage: str
    amount: int
    product_line: str
    close_date: str


class Ticket(BaseModel):
    ticket_id: int
    account_id: int
    subject: str
    severity: str
    status: str
    sla_hours: int
    product_line: str
    opened_at: str


class RMA(BaseModel):
    rma_id: int
    ticket_id: int
    product: str
    serial: str
    status: str


class Partner(BaseModel):
    partner_id: int
    name: str
    region: str
    tier: str


class InventoryItem(BaseModel):
    inventory_id: int
    partner_id: int
    product_line: str
    on_hand: int
    allocated: int
    available: int


class UsageRecord(BaseModel):
    usage_id: int
    account_id: int
    instance_type: str
    gpu_hours: float
    utilization_pct: float
    usage_date: str
```

- [ ] **Step 5: Implement `backend/semantic_layer/apis/store.py`**

```python
"""Materialize the mock-API data once (cached) from the deterministic generators."""

from functools import lru_cache

from semantic_layer.config import settings
from data.generators.api_data import (
    generate_crm,
    generate_itsm,
    generate_partner,
    generate_dgx,
)


@lru_cache
def crm_data():
    return generate_crm(settings.random_seed)


@lru_cache
def itsm_data():
    return generate_itsm(settings.random_seed)


@lru_cache
def partner_data():
    return generate_partner(settings.random_seed)


@lru_cache
def dgx_data():
    return generate_dgx(settings.random_seed)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd backend && ./.venv/bin/python -m pytest tests/test_api_store.py -v`
Expected: PASS (2 passed)

- [ ] **Step 7: Commit**

```bash
git add backend/semantic_layer/apis/__init__.py backend/semantic_layer/apis/models.py backend/semantic_layer/apis/store.py backend/tests/test_api_store.py
git commit -m "feat(apis): pydantic response models and cached data store"
```

---

## Task 3: CRM sub-app

**Files:**
- Create: `backend/semantic_layer/apis/crm.py`
- Test: `backend/tests/test_api_crm.py`

- [ ] **Step 1: Write the failing test** `backend/tests/test_api_crm.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && ./.venv/bin/python -m pytest tests/test_api_crm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'semantic_layer.apis.crm'`

- [ ] **Step 3: Implement `backend/semantic_layer/apis/crm.py`**

```python
"""CRM mock API: accounts, contacts, and sales opportunities (Salesforce-like)."""

from fastapi import FastAPI, HTTPException

from semantic_layer.apis.models import Account, Contact, Opportunity
from semantic_layer.apis.store import crm_data

crm_app = FastAPI(
    title="NVIDIA CRM API",
    version="1.0.0",
    description="Accounts, contacts, and sales opportunities.",
)


@crm_app.get("/accounts", response_model=list[Account])
def list_accounts(region: str | None = None, industry: str | None = None):
    rows = crm_data()["accounts"]
    if region:
        rows = [a for a in rows if a["region"] == region]
    if industry:
        rows = [a for a in rows if a["industry"] == industry]
    return rows


@crm_app.get("/accounts/{account_id}", response_model=Account)
def get_account(account_id: int):
    for a in crm_data()["accounts"]:
        if a["account_id"] == account_id:
            return a
    raise HTTPException(status_code=404, detail="account not found")


@crm_app.get("/opportunities", response_model=list[Opportunity])
def list_opportunities(stage: str | None = None, account_id: int | None = None):
    rows = crm_data()["opportunities"]
    if stage:
        rows = [o for o in rows if o["stage"] == stage]
    if account_id is not None:
        rows = [o for o in rows if o["account_id"] == account_id]
    return rows


@crm_app.get("/contacts", response_model=list[Contact])
def list_contacts(account_id: int | None = None):
    rows = crm_data()["contacts"]
    if account_id is not None:
        rows = [c for c in rows if c["account_id"] == account_id]
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && ./.venv/bin/python -m pytest tests/test_api_crm.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/apis/crm.py backend/tests/test_api_crm.py
git commit -m "feat(apis): CRM sub-app (accounts, contacts, opportunities)"
```

---

## Task 4: Support/ITSM sub-app

**Files:**
- Create: `backend/semantic_layer/apis/itsm.py`
- Test: `backend/tests/test_api_itsm.py`

- [ ] **Step 1: Write the failing test** `backend/tests/test_api_itsm.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && ./.venv/bin/python -m pytest tests/test_api_itsm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'semantic_layer.apis.itsm'`

- [ ] **Step 3: Implement `backend/semantic_layer/apis/itsm.py`**

```python
"""Support/ITSM mock API: tickets and GPU RMAs (ServiceNow-like)."""

from fastapi import FastAPI, HTTPException

from semantic_layer.apis.models import Ticket, RMA
from semantic_layer.apis.store import itsm_data

itsm_app = FastAPI(
    title="NVIDIA Support / ITSM API",
    version="1.0.0",
    description="Support tickets, severities, SLAs, and GPU RMAs.",
)


@itsm_app.get("/tickets", response_model=list[Ticket])
def list_tickets(
    severity: str | None = None,
    status: str | None = None,
    account_id: int | None = None,
):
    rows = itsm_data()["tickets"]
    if severity:
        rows = [t for t in rows if t["severity"] == severity]
    if status:
        rows = [t for t in rows if t["status"] == status]
    if account_id is not None:
        rows = [t for t in rows if t["account_id"] == account_id]
    return rows


@itsm_app.get("/tickets/{ticket_id}", response_model=Ticket)
def get_ticket(ticket_id: int):
    for t in itsm_data()["tickets"]:
        if t["ticket_id"] == ticket_id:
            return t
    raise HTTPException(status_code=404, detail="ticket not found")


@itsm_app.get("/rma", response_model=list[RMA])
def list_rmas(status: str | None = None):
    rows = itsm_data()["rmas"]
    if status:
        rows = [r for r in rows if r["status"] == status]
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && ./.venv/bin/python -m pytest tests/test_api_itsm.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/apis/itsm.py backend/tests/test_api_itsm.py
git commit -m "feat(apis): Support/ITSM sub-app (tickets, RMAs)"
```

---

## Task 5: Partner inventory sub-app

**Files:**
- Create: `backend/semantic_layer/apis/partner.py`
- Test: `backend/tests/test_api_partner.py`

- [ ] **Step 1: Write the failing test** `backend/tests/test_api_partner.py`

```python
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
    assert all(i["partner_id"] == 1 and i["product_line"] == "B200" for i in r.json())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && ./.venv/bin/python -m pytest tests/test_api_partner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'semantic_layer.apis.partner'`

- [ ] **Step 3: Implement `backend/semantic_layer/apis/partner.py`**

```python
"""Partner / channel inventory mock API: distributors and stock levels."""

from fastapi import FastAPI

from semantic_layer.apis.models import Partner, InventoryItem
from semantic_layer.apis.store import partner_data

partner_app = FastAPI(
    title="NVIDIA Partner Inventory API",
    version="1.0.0",
    description="Distributor partners and channel inventory by product line.",
)


@partner_app.get("/partners", response_model=list[Partner])
def list_partners(region: str | None = None):
    rows = partner_data()["partners"]
    if region:
        rows = [p for p in rows if p["region"] == region]
    return rows


@partner_app.get("/inventory", response_model=list[InventoryItem])
def list_inventory(partner_id: int | None = None, product_line: str | None = None):
    rows = partner_data()["inventory"]
    if partner_id is not None:
        rows = [i for i in rows if i["partner_id"] == partner_id]
    if product_line:
        rows = [i for i in rows if i["product_line"] == product_line]
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && ./.venv/bin/python -m pytest tests/test_api_partner.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/apis/partner.py backend/tests/test_api_partner.py
git commit -m "feat(apis): Partner inventory sub-app (partners, inventory)"
```

---

## Task 6: DGX Cloud telemetry sub-app

**Files:**
- Create: `backend/semantic_layer/apis/dgx.py`
- Test: `backend/tests/test_api_dgx.py`

- [ ] **Step 1: Write the failing test** `backend/tests/test_api_dgx.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && ./.venv/bin/python -m pytest tests/test_api_dgx.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'semantic_layer.apis.dgx'`

- [ ] **Step 3: Implement `backend/semantic_layer/apis/dgx.py`**

```python
"""DGX Cloud telemetry mock API: per-account GPU usage time-series."""

from fastapi import FastAPI

from semantic_layer.apis.models import UsageRecord
from semantic_layer.apis.store import dgx_data

dgx_app = FastAPI(
    title="NVIDIA DGX Cloud Telemetry API",
    version="1.0.0",
    description="Per-account GPU-hours, utilization, and instance types.",
)


@dgx_app.get("/usage", response_model=list[UsageRecord])
def list_usage(
    account_id: int | None = None,
    instance_type: str | None = None,
    start: str | None = None,
    end: str | None = None,
):
    rows = dgx_data()["usage"]
    if account_id is not None:
        rows = [u for u in rows if u["account_id"] == account_id]
    if instance_type:
        rows = [u for u in rows if u["instance_type"] == instance_type]
    if start:
        rows = [u for u in rows if u["usage_date"] >= start]
    if end:
        rows = [u for u in rows if u["usage_date"] <= end]
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && ./.venv/bin/python -m pytest tests/test_api_dgx.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/apis/dgx.py backend/tests/test_api_dgx.py
git commit -m "feat(apis): DGX Cloud telemetry sub-app (usage time-series)"
```

---

## Task 7: Main app composition, OpenAPI verification, Makefile + README

**Files:**
- Create: `backend/semantic_layer/apis/app.py`
- Test: `backend/tests/test_api_app.py`
- Modify: `Makefile`
- Modify: `backend/README.md`

- [ ] **Step 1: Write the failing test** `backend/tests/test_api_app.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && ./.venv/bin/python -m pytest tests/test_api_app.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'semantic_layer.apis.app'`

- [ ] **Step 3: Implement `backend/semantic_layer/apis/app.py`**

```python
"""Compose the four mock enterprise APIs as mounted sub-applications.

Each API is mounted under its own path prefix so it exposes an independent
OpenAPI spec at /{prefix}/openapi.json. Run with:
    uvicorn semantic_layer.apis.app:app --port 8001
"""

from fastapi import FastAPI

from semantic_layer.apis.crm import crm_app
from semantic_layer.apis.itsm import itsm_app
from semantic_layer.apis.partner import partner_app
from semantic_layer.apis.dgx import dgx_app

app = FastAPI(title="NVIDIA Enterprise Mock APIs")


@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/crm", crm_app)
app.mount("/itsm", itsm_app)
app.mount("/partner", partner_app)
app.mount("/dgx", dgx_app)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && ./.venv/bin/python -m pytest tests/test_api_app.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Add the `serve-apis` target to `Makefile`**

Add `serve-apis` to the `.PHONY` line, and add this target (recipe line TAB-indented) after the `seed` target:

```makefile
serve-apis:
	cd backend && uvicorn semantic_layer.apis.app:app --port 8001 --reload
```

Verify it parses: run `make -n serve-apis` and confirm it prints the uvicorn command without a "missing separator" error.

- [ ] **Step 6: Document the APIs in `backend/README.md`**

Append this section to the end of `backend/README.md`:

```markdown
## Mock enterprise APIs (Plan 2)

Four NVIDIA-themed mock REST APIs run as one Uvicorn process (mounted sub-apps),
serving deterministic in-memory data. Accounts reuse the sales customer base, so
API records join back to the `sales` data by `account_id` (== `customer_id`).

```bash
source backend/.venv/bin/activate
make serve-apis     # uvicorn on http://localhost:8001
```

| API | Prefix | Key endpoints | OpenAPI |
|-----|--------|---------------|---------|
| CRM | `/crm` | `/accounts`, `/contacts`, `/opportunities` | `/crm/openapi.json` |
| Support/ITSM | `/itsm` | `/tickets`, `/rma` | `/itsm/openapi.json` |
| Partner inventory | `/partner` | `/partners`, `/inventory` | `/partner/openapi.json` |
| DGX Cloud telemetry | `/dgx` | `/usage` | `/dgx/openapi.json` |

Health check: `GET /health`. These OpenAPI specs are what Plan 3's NeoCarta API
extractor introspects into the semantic graph as virtual tables/columns.
```

- [ ] **Step 7: Run the full suite and verify the server boots**

Run: `cd backend && ./.venv/bin/python -m pytest -v`
Expected: all tests pass (Plan 1 tests + the new Plan 2 tests). Paste the summary line. (Postgres-marked tests need the container up; if it is not running they will skip — that is acceptable for this plan since Plan 2 adds no Postgres tests.)

Run a quick boot smoke test:
`cd backend && ./.venv/bin/python -c "from fastapi.testclient import TestClient; from semantic_layer.apis.app import app; c=TestClient(app); print(c.get('/health').json()); print('crm', len(c.get('/crm/accounts').json()), 'itsm', len(c.get('/itsm/tickets').json()), 'partner', len(c.get('/partner/inventory').json()), 'dgx', len(c.get('/dgx/usage').json()))"`
Expected: prints `{'status': 'ok'}` and non-zero counts for all four APIs.

- [ ] **Step 8: Commit**

```bash
git add backend/semantic_layer/apis/app.py backend/tests/test_api_app.py Makefile backend/README.md
git commit -m "feat(apis): mount the four mock APIs as one app; add serve-apis + docs"
```

---

## Self-Review

**Spec coverage (Plan 2 scope):** CRM API (Task 3) ✓ · Support/ITSM API (Task 4) ✓ · Partner inventory API (Task 5) ✓ · DGX Cloud telemetry API (Task 6) ✓ · standalone FastAPI services with their own OpenAPI specs (mounted sub-apps; verified in Task 7) ✓ · synthetic NVIDIA-themed data reusing the sales customer base for cross-source joins (Task 1, `build_accounts`) ✓ · `serve-apis` run target + README (Task 7) ✓. The NeoCarta API extractor that consumes these OpenAPI specs is Plan 3, not here.

**Placeholder scan:** no TBD/TODO/"handle edge cases"; every code step is complete and runnable; every run step has an exact command and expected result.

**Type/name consistency:** generator output keys (`accounts`, `contacts`, `opportunities`; `tickets`, `rmas`; `partners`, `inventory`; `usage`) in `api_data.py` (Task 1) match the `store.py` accessors (Task 2) and every sub-app's `*_data()[...]` lookups (Tasks 3–6). Each row dict's keys match its Pydantic model fields exactly: `Account/Contact/Opportunity` ↔ CRM rows, `Ticket/RMA` ↔ ITSM rows, `Partner/InventoryItem` ↔ partner rows, `UsageRecord` ↔ DGX rows (Task 2 models vs Task 1 dicts). Sub-app symbol names (`crm_app`, `itsm_app`, `partner_app`, `dgx_app`) defined in Tasks 3–6 are imported unchanged by `app.py` (Task 7). Filter query-param names align with the row dict keys used to filter. `account_id == customer_id` linkage (Task 1) is the documented cross-source join key (README, Task 7).

**Scope check:** single coherent subsystem (four small APIs sharing one data layer and one app); appropriately sized for one plan. Independently testable with no DB (TestClient + in-memory data).
