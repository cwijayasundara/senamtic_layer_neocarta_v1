# Plan 1: Data Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the project scaffold, local infrastructure (Neo4j + Postgres in Docker), and fully-seeded NVIDIA-themed synthetic data across a deeply-normalized 11-table Postgres `sales` schema and two SQLite databases — with tests proving the data loads and a 6+-table join returns rows.

**Architecture:** A Python package `semantic_layer` holds config; a `data` package holds deterministic synthetic-data generators (pure functions returning lists of row dicts) and seed scripts that load those rows into Postgres (via `psycopg`) and SQLite (via stdlib `sqlite3`). Generators are seeded with a fixed RNG so output is reproducible and testable. Infrastructure runs via `docker-compose`; a `Makefile` wraps the common commands.

**Tech Stack:** Python 3.11, `psycopg[binary]` (Postgres v3 driver), stdlib `sqlite3`, `pydantic-settings`, `Faker` (seeded), `pytest`; Docker Compose (Postgres 16, Neo4j 5).

This is sub-plan 1 of 5 (Data Foundation → Mock APIs → Graph Ingestion → Agent → Web App). It depends on nothing and produces independently verifiable seeded databases.

---

## File Structure

```
sementic_layer_neocarta_v1/
  docker-compose.yml                      # neo4j + postgres services
  Makefile                                # up · down · seed · test targets
  backend/
    pyproject.toml                        # package + deps + pytest config
    .env.example                          # documented env vars
    semantic_layer/
      __init__.py
      config.py                           # pydantic-settings Settings
    data/
      __init__.py
      generators/
        __init__.py
        reference.py                      # curated NVIDIA dimensions + id assignment
        sales.py                          # products, customers, orders, order_lines
        financials.py                     # income statement + stock prices
        org.py                            # departments, locations, headcount
      schema/
        postgres_sales.sql                # DDL for the 11 sales tables
      seed_postgres.py                    # create schema + load sales rows into Postgres
      seed_sqlite.py                      # build financials.db and org.db
    tests/
      __init__.py
      conftest.py                         # postgres availability fixture
      test_config.py
      test_reference.py
      test_sales_generator.py
      test_financials_generator.py
      test_org_generator.py
      test_seed_postgres.py               # integration: deep 6+-table join
      test_seed_sqlite.py
```

**Responsibilities:** generators are pure (no I/O) so they unit-test without a database. Seed scripts are the only I/O boundary. `config.py` is the single source of connection settings.

---

## Task 1: Project scaffold and configuration

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/.env.example`
- Create: `backend/semantic_layer/__init__.py`
- Create: `backend/semantic_layer/config.py`
- Create: `backend/data/__init__.py`
- Create: `backend/data/generators/__init__.py`
- Create: `backend/tests/__init__.py`
- Test: `backend/tests/test_config.py`

- [ ] **Step 1: Create `backend/pyproject.toml`**

```toml
[project]
name = "semantic-layer"
version = "0.1.0"
description = "NeoCarta-Local semantic layer over databases, APIs, and documents"
requires-python = ">=3.11"
dependencies = [
    "pydantic-settings>=2.2",
    "psycopg[binary]>=3.1",
    "Faker>=25.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["semantic_layer*", "data*"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
markers = ["postgres: integration tests that require the docker postgres service"]
```

- [ ] **Step 2: Create `backend/.env.example`**

```bash
# Postgres (matches docker-compose.yml)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=nvidia
POSTGRES_USER=neocarta
POSTGRES_PASSWORD=neocarta

# SQLite output directory (relative to backend/)
SQLITE_DIR=data/seed

# Neo4j (used by later plans)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neocarta123

# OpenAI (used by later plans)
OPENAI_API_KEY=

# Deterministic synthetic data
RANDOM_SEED=42
```

- [ ] **Step 3: Create the empty package markers**

Create `backend/semantic_layer/__init__.py`, `backend/data/__init__.py`, `backend/data/generators/__init__.py`, and `backend/tests/__init__.py` each as an empty file:

```python
```

- [ ] **Step 4: Write the failing test** `backend/tests/test_config.py`

```python
from semantic_layer.config import Settings


def test_defaults_load_without_env_file():
    s = Settings(_env_file=None)
    assert s.postgres_db == "nvidia"
    assert s.postgres_port == 5432
    assert s.random_seed == 42
    assert s.sqlite_dir == "data/seed"


def test_postgres_dsn_is_well_formed():
    s = Settings(_env_file=None)
    assert s.postgres_dsn == (
        "postgresql://neocarta:neocarta@localhost:5432/nvidia"
    )
```

- [ ] **Step 5: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'semantic_layer.config'`

- [ ] **Step 6: Implement `backend/semantic_layer/config.py`**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "nvidia"
    postgres_user: str = "neocarta"
    postgres_password: str = "neocarta"

    sqlite_dir: str = "data/seed"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neocarta123"

    openai_api_key: str = ""

    random_seed: int = 42

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 8: Commit**

```bash
git add backend/pyproject.toml backend/.env.example backend/semantic_layer backend/data backend/tests
git commit -m "feat(scaffold): project package, config settings, and pyproject"
```

---

## Task 2: Docker infrastructure (Neo4j + Postgres) and Makefile

**Files:**
- Create: `docker-compose.yml`
- Create: `Makefile`

- [ ] **Step 1: Create `docker-compose.yml`**

```yaml
services:
  postgres:
    image: postgres:16
    container_name: neocarta-postgres
    environment:
      POSTGRES_DB: nvidia
      POSTGRES_USER: neocarta
      POSTGRES_PASSWORD: neocarta
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U neocarta -d nvidia"]
      interval: 5s
      timeout: 5s
      retries: 10
    volumes:
      - pgdata:/var/lib/postgresql/data

  neo4j:
    image: neo4j:5
    container_name: neocarta-neo4j
    environment:
      NEO4J_AUTH: neo4j/neocarta123
      NEO4J_PLUGINS: '["apoc"]'
    ports:
      - "7474:7474"
      - "7687:7687"
    healthcheck:
      test: ["CMD-SHELL", "wget -qO- http://localhost:7474 || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 10
    volumes:
      - neo4jdata:/data

volumes:
  pgdata:
  neo4jdata:
```

- [ ] **Step 2: Create `Makefile`**

```makefile
.PHONY: up down seed seed-postgres seed-sqlite test install

install:
	cd backend && pip install -e ".[dev]"

up:
	docker compose up -d
	@echo "Waiting for Postgres to be healthy..."
	@until docker inspect --format '{{.State.Health.Status}}' neocarta-postgres | grep -q healthy; do sleep 2; done
	@echo "Postgres ready."

down:
	docker compose down

seed-postgres:
	cd backend && python -m data.seed_postgres

seed-sqlite:
	cd backend && python -m data.seed_sqlite

seed: seed-postgres seed-sqlite

test:
	cd backend && python -m pytest -v
```

- [ ] **Step 3: Verify infrastructure starts**

Run: `make up`
Expected: containers `neocarta-postgres` and `neocarta-neo4j` start; command prints `Postgres ready.`

Run: `docker exec neocarta-postgres psql -U neocarta -d nvidia -c "SELECT 1;"`
Expected: returns a row with `1`.

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml Makefile
git commit -m "feat(infra): docker-compose for postgres + neo4j and Makefile targets"
```

---

## Task 3: Curated reference dimensions

**Files:**
- Create: `backend/data/generators/reference.py`
- Test: `backend/tests/test_reference.py`

- [ ] **Step 1: Write the failing test** `backend/tests/test_reference.py`

```python
from data.generators.reference import build_dimensions


def test_dimensions_have_expected_counts():
    d = build_dimensions()
    assert len(d["regions"]) == 4
    assert len(d["countries"]) == 9
    assert len(d["industries"]) == 6
    assert len(d["segments"]) == 5
    assert len(d["architectures"]) == 5
    assert len(d["product_lines"]) == 10
    assert len(d["fiscal_periods"]) == 8


def test_every_country_points_to_a_valid_region():
    d = build_dimensions()
    region_ids = {r["region_id"] for r in d["regions"]}
    assert all(c["region_id"] in region_ids for c in d["countries"])


def test_product_lines_reference_valid_segment_and_architecture():
    d = build_dimensions()
    seg_ids = {s["segment_id"] for s in d["segments"]}
    arch_ids = {a["architecture_id"] for a in d["architectures"]}
    for pl in d["product_lines"]:
        assert pl["segment_id"] in seg_ids
        assert pl["architecture_id"] in arch_ids


def test_blackwell_data_center_line_exists():
    d = build_dimensions()
    seg = {s["segment_id"]: s["name"] for s in d["segments"]}
    arch = {a["architecture_id"]: a["name"] for a in d["architectures"]}
    pairs = {(seg[pl["segment_id"]], arch[pl["architecture_id"]]) for pl in d["product_lines"]}
    assert ("Data Center", "Blackwell") in pairs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_reference.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.generators.reference'`

- [ ] **Step 3: Implement `backend/data/generators/reference.py`**

```python
"""Curated NVIDIA-themed dimension data with deterministic integer ids."""

REGIONS = ["North America", "EMEA", "APAC", "LATAM"]

# (name, iso_code, region_name)
COUNTRIES = [
    ("United States", "US", "North America"),
    ("Canada", "CA", "North America"),
    ("United Kingdom", "GB", "EMEA"),
    ("Germany", "DE", "EMEA"),
    ("United Arab Emirates", "AE", "EMEA"),
    ("Japan", "JP", "APAC"),
    ("South Korea", "KR", "APAC"),
    ("Singapore", "SG", "APAC"),
    ("Brazil", "BR", "LATAM"),
]

INDUSTRIES = [
    "Cloud Service Provider",
    "Automotive",
    "Healthcare",
    "Gaming",
    "Research",
    "Financial Services",
]

SEGMENTS = [
    "Data Center",
    "Gaming",
    "Professional Visualization",
    "Automotive",
    "OEM & Other",
]

# (name, launch_year)
ARCHITECTURES = [
    ("Ampere", 2020),
    ("Ada Lovelace", 2022),
    ("Hopper", 2022),
    ("Grace", 2023),
    ("Blackwell", 2024),
]

# (name, segment_name, architecture_name)
PRODUCT_LINES = [
    ("H100", "Data Center", "Hopper"),
    ("H200", "Data Center", "Hopper"),
    ("B200", "Data Center", "Blackwell"),
    ("GB200", "Data Center", "Blackwell"),
    ("A100", "Data Center", "Ampere"),
    ("RTX 4090", "Gaming", "Ada Lovelace"),
    ("RTX 4080", "Gaming", "Ada Lovelace"),
    ("RTX 5090", "Gaming", "Blackwell"),
    ("RTX 6000 Ada", "Professional Visualization", "Ada Lovelace"),
    ("DRIVE Thor", "Automotive", "Blackwell"),
]

# NVIDIA fiscal year ends late January; FY label = the calendar year it ends in.
# (fiscal_year, quarter, start_date, end_date)
FISCAL_PERIODS = [
    (2024, "Q1", "2023-02-01", "2023-04-30"),
    (2024, "Q2", "2023-05-01", "2023-07-31"),
    (2024, "Q3", "2023-08-01", "2023-10-31"),
    (2024, "Q4", "2023-11-01", "2024-01-28"),
    (2025, "Q1", "2024-01-29", "2024-04-28"),
    (2025, "Q2", "2024-04-29", "2024-07-28"),
    (2025, "Q3", "2024-07-29", "2024-10-27"),
    (2025, "Q4", "2024-10-28", "2025-01-26"),
]


def build_dimensions():
    """Return all dimension rows with assigned integer primary keys and FKs."""
    regions = [{"region_id": i + 1, "name": n} for i, n in enumerate(REGIONS)]
    region_id = {r["name"]: r["region_id"] for r in regions}

    countries = [
        {"country_id": i + 1, "name": n, "iso_code": iso, "region_id": region_id[reg]}
        for i, (n, iso, reg) in enumerate(COUNTRIES)
    ]

    industries = [{"industry_id": i + 1, "name": n} for i, n in enumerate(INDUSTRIES)]
    segments = [{"segment_id": i + 1, "name": n} for i, n in enumerate(SEGMENTS)]
    architectures = [
        {"architecture_id": i + 1, "name": n, "launch_year": y}
        for i, (n, y) in enumerate(ARCHITECTURES)
    ]

    seg_id = {s["name"]: s["segment_id"] for s in segments}
    arch_id = {a["name"]: a["architecture_id"] for a in architectures}
    product_lines = [
        {
            "product_line_id": i + 1,
            "name": n,
            "segment_id": seg_id[seg],
            "architecture_id": arch_id[arch],
        }
        for i, (n, seg, arch) in enumerate(PRODUCT_LINES)
    ]

    fiscal_periods = [
        {
            "fiscal_period_id": i + 1,
            "fiscal_year": fy,
            "quarter": q,
            "start_date": s,
            "end_date": e,
        }
        for i, (fy, q, s, e) in enumerate(FISCAL_PERIODS)
    ]

    return {
        "regions": regions,
        "countries": countries,
        "industries": industries,
        "segments": segments,
        "architectures": architectures,
        "product_lines": product_lines,
        "fiscal_periods": fiscal_periods,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_reference.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/data/generators/reference.py backend/tests/test_reference.py
git commit -m "feat(data): curated NVIDIA reference dimensions"
```

---

## Task 4: Sales fact generator (products, customers, orders, order lines)

**Files:**
- Create: `backend/data/generators/sales.py`
- Test: `backend/tests/test_sales_generator.py`

- [ ] **Step 1: Write the failing test** `backend/tests/test_sales_generator.py`

```python
from data.generators.sales import generate_sales


def test_generation_is_deterministic():
    a = generate_sales(seed=42)
    b = generate_sales(seed=42)
    assert a["order_lines"] == b["order_lines"]


def test_referential_integrity_products_to_product_lines():
    data = generate_sales(seed=42)
    pl_ids = {pl["product_line_id"] for pl in data["product_lines"]}
    assert all(p["product_line_id"] in pl_ids for p in data["products"])


def test_order_lines_reference_valid_orders_and_products():
    data = generate_sales(seed=42)
    order_ids = {o["order_id"] for o in data["sales_orders"]}
    product_ids = {p["product_id"] for p in data["products"]}
    for ol in data["order_lines"]:
        assert ol["order_id"] in order_ids
        assert ol["product_id"] in product_ids


def test_amount_equals_quantity_times_unit_price():
    data = generate_sales(seed=42)
    for ol in data["order_lines"]:
        assert ol["amount"] == round(ol["quantity"] * ol["unit_price"], 2)


def test_volumes_are_reasonable():
    data = generate_sales(seed=42, n_customers=40, n_orders=300)
    assert len(data["products"]) == 20          # 2 variants per product line
    assert len(data["customers"]) == 40
    assert len(data["sales_orders"]) == 300
    assert len(data["order_lines"]) >= 300
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_sales_generator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.generators.sales'`

- [ ] **Step 3: Implement `backend/data/generators/sales.py`**

```python
"""Deterministic synthetic sales facts built on the curated dimensions."""

import random

from faker import Faker

from .reference import build_dimensions

_VARIANTS = ["", " SXM"]


def generate_sales(seed=42, n_customers=40, n_orders=300):
    rng = random.Random(seed)
    fake = Faker()
    fake.seed_instance(seed)

    dims = build_dimensions()

    # --- products: two variants per product line ---
    products = []
    pid = 0
    dc_segment_id = next(s["segment_id"] for s in dims["segments"] if s["name"] == "Data Center")
    seg_by_line = {pl["product_line_id"]: pl["segment_id"] for pl in dims["product_lines"]}
    for pl in dims["product_lines"]:
        for variant in _VARIANTS:
            pid += 1
            base = 28000 if seg_by_line[pl["product_line_id"]] == dc_segment_id else 1500
            msrp = base + rng.randint(0, 12000)
            products.append(
                {
                    "product_id": pid,
                    "product_line_id": pl["product_line_id"],
                    "sku": f"NV-{pl['name'].replace(' ', '')}-{pid:03d}",
                    "name": (pl["name"] + variant).strip(),
                    "msrp": msrp,
                    "launch_date": f"{2020 + rng.randint(0, 4)}-{rng.randint(1, 12):02d}-15",
                }
            )

    # --- customers ---
    country_ids = [c["country_id"] for c in dims["countries"]]
    industry_ids = [i["industry_id"] for i in dims["industries"]]
    customers = []
    for cid in range(1, n_customers + 1):
        customers.append(
            {
                "customer_id": cid,
                "name": fake.unique.company(),
                "country_id": rng.choice(country_ids),
                "industry_id": rng.choice(industry_ids),
            }
        )

    # --- orders ---
    customer_ids = [c["customer_id"] for c in customers]
    fiscal_period_ids = [fp["fiscal_period_id"] for fp in dims["fiscal_periods"]]
    fp_by_id = {fp["fiscal_period_id"]: fp for fp in dims["fiscal_periods"]}
    sales_orders = []
    for oid in range(1, n_orders + 1):
        fpid = rng.choice(fiscal_period_ids)
        sales_orders.append(
            {
                "order_id": oid,
                "customer_id": rng.choice(customer_ids),
                "fiscal_period_id": fpid,
                "order_date": fp_by_id[fpid]["start_date"],
            }
        )

    # --- order lines: 1-3 per order ---
    product_index = [p["product_id"] for p in products]
    msrp_by_product = {p["product_id"]: p["msrp"] for p in products}
    order_lines = []
    lid = 0
    for o in sales_orders:
        for _ in range(rng.randint(1, 3)):
            lid += 1
            prod = rng.choice(product_index)
            qty = rng.randint(1, 64)
            unit_price = round(msrp_by_product[prod] * rng.uniform(0.85, 1.0), 2)
            order_lines.append(
                {
                    "line_id": lid,
                    "order_id": o["order_id"],
                    "product_id": prod,
                    "quantity": qty,
                    "unit_price": unit_price,
                    "amount": round(qty * unit_price, 2),
                }
            )

    return {**dims, "products": products, "customers": customers,
            "sales_orders": sales_orders, "order_lines": order_lines}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_sales_generator.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/data/generators/sales.py backend/tests/test_sales_generator.py
git commit -m "feat(data): deterministic sales fact generator"
```

---

## Task 5: Financials generator (SQLite source data)

**Files:**
- Create: `backend/data/generators/financials.py`
- Test: `backend/tests/test_financials_generator.py`

- [ ] **Step 1: Write the failing test** `backend/tests/test_financials_generator.py`

```python
from data.generators.financials import generate_financials


def test_income_statement_one_row_per_fiscal_quarter():
    data = generate_financials(seed=42)
    keys = {(r["fiscal_year"], r["quarter"]) for r in data["income_statement"]}
    assert len(keys) == len(data["income_statement"]) == 8


def test_net_income_not_above_revenue():
    data = generate_financials(seed=42)
    for r in data["income_statement"]:
        assert 0 < r["net_income"] <= r["revenue"]


def test_stock_prices_are_ohlc_consistent():
    data = generate_financials(seed=42)
    assert len(data["stock_prices"]) > 0
    for p in data["stock_prices"]:
        assert p["low"] <= p["open"] <= p["high"]
        assert p["low"] <= p["close"] <= p["high"]


def test_deterministic():
    assert generate_financials(seed=42) == generate_financials(seed=42)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_financials_generator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.generators.financials'`

- [ ] **Step 3: Implement `backend/data/generators/financials.py`**

```python
"""Deterministic quarterly income statement and daily stock prices."""

import random
from datetime import date, timedelta

from .reference import FISCAL_PERIODS


def generate_financials(seed=42):
    rng = random.Random(seed)

    income_statement = []
    pid = 0
    revenue = 7000.0  # $M, grows each quarter
    for fy, q, start, end in FISCAL_PERIODS:
        pid += 1
        revenue = round(revenue * rng.uniform(1.05, 1.30), 1)
        gross_margin_pct = round(rng.uniform(60.0, 75.0), 1)
        operating_income = round(revenue * rng.uniform(0.30, 0.55), 1)
        net_income = round(operating_income * rng.uniform(0.80, 0.95), 1)
        eps = round(net_income / 2470.0, 2)  # ~2.47B diluted shares
        income_statement.append(
            {
                "period_id": pid,
                "fiscal_year": fy,
                "quarter": q,
                "revenue": revenue,
                "gross_margin_pct": gross_margin_pct,
                "operating_income": operating_income,
                "net_income": net_income,
                "eps": eps,
            }
        )

    # daily stock prices across the whole fiscal range
    start_date = date.fromisoformat(FISCAL_PERIODS[0][2])
    end_date = date.fromisoformat(FISCAL_PERIODS[-1][3])
    stock_prices = []
    price = 22.0
    d = start_date
    while d <= end_date:
        if d.weekday() < 5:  # weekdays only
            change = rng.uniform(-0.04, 0.05)
            open_p = round(price, 2)
            close_p = round(price * (1 + change), 2)
            high_p = round(max(open_p, close_p) * rng.uniform(1.0, 1.03), 2)
            low_p = round(min(open_p, close_p) * rng.uniform(0.97, 1.0), 2)
            stock_prices.append(
                {
                    "price_date": d.isoformat(),
                    "open": open_p,
                    "high": high_p,
                    "low": low_p,
                    "close": close_p,
                    "volume": rng.randint(200_000_000, 600_000_000),
                }
            )
            price = close_p
        d += timedelta(days=1)

    return {"income_statement": income_statement, "stock_prices": stock_prices}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_financials_generator.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/data/generators/financials.py backend/tests/test_financials_generator.py
git commit -m "feat(data): financials generator (income statement + stock prices)"
```

---

## Task 6: Org generator (SQLite source data)

**Files:**
- Create: `backend/data/generators/org.py`
- Test: `backend/tests/test_org_generator.py`

- [ ] **Step 1: Write the failing test** `backend/tests/test_org_generator.py`

```python
from data.generators.org import generate_org


def test_departments_and_locations_present():
    data = generate_org(seed=42)
    assert len(data["departments"]) == 6
    assert len(data["locations"]) == 5


def test_headcount_references_valid_dimensions():
    data = generate_org(seed=42)
    dept_ids = {d["department_id"] for d in data["departments"]}
    loc_ids = {l["location_id"] for l in data["locations"]}
    for h in data["headcount"]:
        assert h["department_id"] in dept_ids
        assert h["location_id"] in loc_ids
        assert h["employee_count"] > 0


def test_deterministic():
    assert generate_org(seed=42) == generate_org(seed=42)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_org_generator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.generators.org'`

- [ ] **Step 3: Implement `backend/data/generators/org.py`**

```python
"""Deterministic org / headcount data for the org.db SQLite source."""

import random

DEPARTMENTS = [
    "Research & Development",
    "Sales & Marketing",
    "Operations",
    "Finance",
    "Legal",
    "Information Technology",
]

# (city, country, region)
LOCATIONS = [
    ("Santa Clara", "United States", "North America"),
    ("Austin", "United States", "North America"),
    ("Tel Aviv", "Israel", "EMEA"),
    ("Bangalore", "India", "APAC"),
    ("Taipei", "Taiwan", "APAC"),
]

_FISCAL = [(2024, q) for q in ("Q1", "Q2", "Q3", "Q4")] + [
    (2025, q) for q in ("Q1", "Q2", "Q3", "Q4")
]


def generate_org(seed=42):
    rng = random.Random(seed)

    departments = [
        {"department_id": i + 1, "name": n} for i, n in enumerate(DEPARTMENTS)
    ]
    locations = [
        {"location_id": i + 1, "city": c, "country": co, "region": reg}
        for i, (c, co, reg) in enumerate(LOCATIONS)
    ]

    headcount = []
    sid = 0
    for dept in departments:
        for loc in locations:
            base = rng.randint(50, 1200)
            for fy, q in _FISCAL:
                sid += 1
                base = int(base * rng.uniform(1.0, 1.08))  # grows over time
                headcount.append(
                    {
                        "snapshot_id": sid,
                        "department_id": dept["department_id"],
                        "location_id": loc["location_id"],
                        "fiscal_year": fy,
                        "quarter": q,
                        "employee_count": base,
                    }
                )

    return {"departments": departments, "locations": locations, "headcount": headcount}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_org_generator.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/data/generators/org.py backend/tests/test_org_generator.py
git commit -m "feat(data): org / headcount generator"
```

---

## Task 7: Postgres schema DDL and seed script

**Files:**
- Create: `backend/data/schema/postgres_sales.sql`
- Create: `backend/data/seed_postgres.py`
- Create: `backend/tests/conftest.py`
- Test: `backend/tests/test_seed_postgres.py`

- [ ] **Step 1: Create `backend/data/schema/postgres_sales.sql`**

```sql
DROP SCHEMA IF EXISTS sales CASCADE;
CREATE SCHEMA sales;

CREATE TABLE sales.region (
    region_id   INTEGER PRIMARY KEY,
    name        TEXT NOT NULL
);

CREATE TABLE sales.country (
    country_id  INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    iso_code    TEXT NOT NULL,
    region_id   INTEGER NOT NULL REFERENCES sales.region(region_id)
);

CREATE TABLE sales.industry (
    industry_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL
);

CREATE TABLE sales.customer (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    country_id  INTEGER NOT NULL REFERENCES sales.country(country_id),
    industry_id INTEGER NOT NULL REFERENCES sales.industry(industry_id)
);

CREATE TABLE sales.segment (
    segment_id  INTEGER PRIMARY KEY,
    name        TEXT NOT NULL
);

CREATE TABLE sales.architecture (
    architecture_id INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    launch_year     INTEGER NOT NULL
);

CREATE TABLE sales.product_line (
    product_line_id INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    segment_id      INTEGER NOT NULL REFERENCES sales.segment(segment_id),
    architecture_id INTEGER NOT NULL REFERENCES sales.architecture(architecture_id)
);

CREATE TABLE sales.product (
    product_id      INTEGER PRIMARY KEY,
    product_line_id INTEGER NOT NULL REFERENCES sales.product_line(product_line_id),
    sku             TEXT NOT NULL,
    name            TEXT NOT NULL,
    msrp            NUMERIC(12,2) NOT NULL,
    launch_date     DATE NOT NULL
);

CREATE TABLE sales.fiscal_period (
    fiscal_period_id INTEGER PRIMARY KEY,
    fiscal_year      INTEGER NOT NULL,
    quarter          TEXT NOT NULL,
    start_date       DATE NOT NULL,
    end_date         DATE NOT NULL
);

CREATE TABLE sales.sales_order (
    order_id         INTEGER PRIMARY KEY,
    customer_id      INTEGER NOT NULL REFERENCES sales.customer(customer_id),
    fiscal_period_id INTEGER NOT NULL REFERENCES sales.fiscal_period(fiscal_period_id),
    order_date       DATE NOT NULL
);

CREATE TABLE sales.order_line (
    line_id    INTEGER PRIMARY KEY,
    order_id   INTEGER NOT NULL REFERENCES sales.sales_order(order_id),
    product_id INTEGER NOT NULL REFERENCES sales.product(product_id),
    quantity   INTEGER NOT NULL,
    unit_price NUMERIC(12,2) NOT NULL,
    amount     NUMERIC(14,2) NOT NULL
);
```

- [ ] **Step 2: Implement `backend/data/seed_postgres.py`**

```python
"""Create the sales schema and load generated rows into Postgres."""

from pathlib import Path

import psycopg

from semantic_layer.config import settings
from data.generators.sales import generate_sales

_SCHEMA_FILE = Path(__file__).parent / "schema" / "postgres_sales.sql"

# table name -> ordered column list (insertion order matches FK dependencies)
_TABLES = [
    ("sales.region", ["region_id", "name"]),
    ("sales.country", ["country_id", "name", "iso_code", "region_id"]),
    ("sales.industry", ["industry_id", "name"]),
    ("sales.customer", ["customer_id", "name", "country_id", "industry_id"]),
    ("sales.segment", ["segment_id", "name"]),
    ("sales.architecture", ["architecture_id", "name", "launch_year"]),
    ("sales.product_line", ["product_line_id", "name", "segment_id", "architecture_id"]),
    ("sales.product", ["product_id", "product_line_id", "sku", "name", "msrp", "launch_date"]),
    ("sales.fiscal_period", ["fiscal_period_id", "fiscal_year", "quarter", "start_date", "end_date"]),
    ("sales.sales_order", ["order_id", "customer_id", "fiscal_period_id", "order_date"]),
    ("sales.order_line", ["line_id", "order_id", "product_id", "quantity", "unit_price", "amount"]),
]

# data key per table (the generate_sales() output uses these keys)
_DATA_KEY = {
    "sales.region": "regions",
    "sales.country": "countries",
    "sales.industry": "industries",
    "sales.customer": "customers",
    "sales.segment": "segments",
    "sales.architecture": "architectures",
    "sales.product_line": "product_lines",
    "sales.product": "products",
    "sales.fiscal_period": "fiscal_periods",
    "sales.sales_order": "sales_orders",
    "sales.order_line": "order_lines",
}


def seed(dsn: str | None = None, seed_value: int | None = None) -> dict:
    dsn = dsn or settings.postgres_dsn
    data = generate_sales(seed=seed_value if seed_value is not None else settings.random_seed)
    counts = {}
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA_FILE.read_text())
            for table, cols in _TABLES:
                rows = data[_DATA_KEY[table]]
                placeholders = ", ".join(["%s"] * len(cols))
                stmt = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
                cur.executemany(stmt, [[r[c] for c in cols] for r in rows])
                counts[table] = len(rows)
        conn.commit()
    return counts


if __name__ == "__main__":
    result = seed()
    for table, n in result.items():
        print(f"{table}: {n} rows")
```

- [ ] **Step 3: Create `backend/tests/conftest.py`**

```python
import psycopg
import pytest

from semantic_layer.config import settings


@pytest.fixture(scope="session")
def postgres_dsn():
    """Skip postgres-marked tests if the docker postgres is not reachable."""
    dsn = settings.postgres_dsn
    try:
        with psycopg.connect(dsn, connect_timeout=3):
            pass
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres not available at {dsn}: {exc}")
    return dsn
```

- [ ] **Step 4: Write the failing test** `backend/tests/test_seed_postgres.py`

```python
import psycopg
import pytest

from data.seed_postgres import seed

DEEP_JOIN_SQL = """
SELECT COUNT(*) AS line_count, COALESCE(SUM(ol.amount), 0) AS revenue
FROM sales.order_line ol
JOIN sales.product p           ON p.product_id = ol.product_id
JOIN sales.product_line pl     ON pl.product_line_id = p.product_line_id
JOIN sales.segment s           ON s.segment_id = pl.segment_id
JOIN sales.architecture a      ON a.architecture_id = pl.architecture_id
JOIN sales.sales_order so      ON so.order_id = ol.order_id
JOIN sales.fiscal_period fp    ON fp.fiscal_period_id = so.fiscal_period_id
JOIN sales.customer c          ON c.customer_id = so.customer_id
JOIN sales.industry i          ON i.industry_id = c.industry_id
JOIN sales.country co          ON co.country_id = c.country_id
JOIN sales.region r            ON r.region_id = co.region_id
WHERE s.name = 'Data Center' AND a.name = 'Blackwell';
"""


@pytest.mark.postgres
def test_seed_loads_all_tables(postgres_dsn):
    counts = seed(dsn=postgres_dsn)
    assert counts["sales.order_line"] >= 300
    with psycopg.connect(postgres_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM sales.product;")
        assert cur.fetchone()[0] == 20


@pytest.mark.postgres
def test_deep_eleven_table_join_returns_rows(postgres_dsn):
    seed(dsn=postgres_dsn)
    with psycopg.connect(postgres_dsn) as conn, conn.cursor() as cur:
        cur.execute(DEEP_JOIN_SQL)
        line_count, revenue = cur.fetchone()
    assert line_count > 0
    assert revenue > 0
```

- [ ] **Step 5: Run test to verify it fails (or skips without DB), then passes with DB**

Run: `make up` then `cd backend && python -m pytest tests/test_seed_postgres.py -v`
Expected: FAIL first with `ModuleNotFoundError: No module named 'data.seed_postgres'` before Step 2 exists; after implementing, PASS (2 passed). If Postgres is down, tests SKIP with the conftest message.

- [ ] **Step 6: Commit**

```bash
git add backend/data/schema/postgres_sales.sql backend/data/seed_postgres.py backend/tests/conftest.py backend/tests/test_seed_postgres.py
git commit -m "feat(data): postgres sales schema + seed with 11-table deep-join test"
```

---

## Task 8: SQLite seed script (financials.db + org.db)

**Files:**
- Create: `backend/data/seed_sqlite.py`
- Test: `backend/tests/test_seed_sqlite.py`

- [ ] **Step 1: Write the failing test** `backend/tests/test_seed_sqlite.py`

```python
import sqlite3

from data.seed_sqlite import seed_all


def test_seed_creates_both_databases(tmp_path):
    paths = seed_all(out_dir=str(tmp_path))
    assert (tmp_path / "financials.db").exists()
    assert (tmp_path / "org.db").exists()
    assert set(paths) == {"financials", "org"}


def test_financials_tables_populated(tmp_path):
    seed_all(out_dir=str(tmp_path))
    con = sqlite3.connect(tmp_path / "financials.db")
    assert con.execute("SELECT COUNT(*) FROM income_statement").fetchone()[0] == 8
    assert con.execute("SELECT COUNT(*) FROM stock_price").fetchone()[0] > 0
    con.close()


def test_org_join_returns_headcount_by_region(tmp_path):
    seed_all(out_dir=str(tmp_path))
    con = sqlite3.connect(tmp_path / "org.db")
    rows = con.execute(
        """
        SELECT l.region, SUM(h.employee_count)
        FROM headcount h
        JOIN location l ON l.location_id = h.location_id
        JOIN department d ON d.department_id = h.department_id
        WHERE h.fiscal_year = 2025 AND h.quarter = 'Q4'
        GROUP BY l.region
        """
    ).fetchall()
    con.close()
    assert len(rows) > 0
    assert all(total > 0 for _, total in rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_seed_sqlite.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.seed_sqlite'`

- [ ] **Step 3: Implement `backend/data/seed_sqlite.py`**

```python
"""Build the financials.db and org.db SQLite source databases."""

import sqlite3
from pathlib import Path

from semantic_layer.config import settings
from data.generators.financials import generate_financials
from data.generators.org import generate_org

_FINANCIALS_DDL = """
DROP TABLE IF EXISTS income_statement;
CREATE TABLE income_statement (
    period_id        INTEGER PRIMARY KEY,
    fiscal_year      INTEGER NOT NULL,
    quarter          TEXT NOT NULL,
    revenue          REAL NOT NULL,
    gross_margin_pct REAL NOT NULL,
    operating_income REAL NOT NULL,
    net_income       REAL NOT NULL,
    eps              REAL NOT NULL
);
DROP TABLE IF EXISTS stock_price;
CREATE TABLE stock_price (
    price_date TEXT PRIMARY KEY,
    open       REAL NOT NULL,
    high       REAL NOT NULL,
    low        REAL NOT NULL,
    close      REAL NOT NULL,
    volume     INTEGER NOT NULL
);
"""

_ORG_DDL = """
DROP TABLE IF EXISTS department;
CREATE TABLE department (
    department_id INTEGER PRIMARY KEY,
    name          TEXT NOT NULL
);
DROP TABLE IF EXISTS location;
CREATE TABLE location (
    location_id INTEGER PRIMARY KEY,
    city        TEXT NOT NULL,
    country     TEXT NOT NULL,
    region      TEXT NOT NULL
);
DROP TABLE IF EXISTS headcount;
CREATE TABLE headcount (
    snapshot_id    INTEGER PRIMARY KEY,
    department_id  INTEGER NOT NULL REFERENCES department(department_id),
    location_id    INTEGER NOT NULL REFERENCES location(location_id),
    fiscal_year    INTEGER NOT NULL,
    quarter        TEXT NOT NULL,
    employee_count INTEGER NOT NULL
);
"""


def _insert(con, table, cols, rows):
    placeholders = ", ".join(["?"] * len(cols))
    con.executemany(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})",
        [[r[c] for c in cols] for r in rows],
    )


def seed_all(out_dir: str | None = None, seed_value: int | None = None) -> dict:
    out = Path(out_dir or settings.sqlite_dir)
    out.mkdir(parents=True, exist_ok=True)
    sv = seed_value if seed_value is not None else settings.random_seed

    fin = generate_financials(seed=sv)
    fin_path = out / "financials.db"
    con = sqlite3.connect(fin_path)
    con.executescript(_FINANCIALS_DDL)
    _insert(con, "income_statement",
            ["period_id", "fiscal_year", "quarter", "revenue", "gross_margin_pct",
             "operating_income", "net_income", "eps"], fin["income_statement"])
    _insert(con, "stock_price",
            ["price_date", "open", "high", "low", "close", "volume"], fin["stock_prices"])
    con.commit()
    con.close()

    org = generate_org(seed=sv)
    org_path = out / "org.db"
    con = sqlite3.connect(org_path)
    con.executescript(_ORG_DDL)
    _insert(con, "department", ["department_id", "name"], org["departments"])
    _insert(con, "location", ["location_id", "city", "country", "region"], org["locations"])
    _insert(con, "headcount",
            ["snapshot_id", "department_id", "location_id", "fiscal_year",
             "quarter", "employee_count"], org["headcount"])
    con.commit()
    con.close()

    return {"financials": str(fin_path), "org": str(org_path)}


if __name__ == "__main__":
    result = seed_all()
    for name, path in result.items():
        print(f"{name}: {path}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_seed_sqlite.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/data/seed_sqlite.py backend/tests/test_seed_sqlite.py
git commit -m "feat(data): sqlite seed for financials.db and org.db"
```

---

## Task 9: Full-suite verification and README note

**Files:**
- Create: `backend/README.md`

- [ ] **Step 1: Run the entire test suite with infrastructure up**

Run: `make up && make install && make test`
Expected: all tests PASS (postgres-marked tests run, not skipped). If any fail, fix before continuing.

- [ ] **Step 2: Run the seed end-to-end**

Run: `make seed`
Expected: prints per-table row counts for Postgres and the two SQLite file paths. `backend/data/seed/financials.db` and `backend/data/seed/org.db` exist.

- [ ] **Step 3: Create `backend/README.md`**

```markdown
# NeoCarta-Local — Backend (Plan 1: Data Foundation)

## Prerequisites
- Docker + Docker Compose
- Python 3.11+

## Setup
```bash
cp backend/.env.example backend/.env   # adjust if needed
make up         # start postgres + neo4j
make install    # pip install -e ".[dev]"
make seed       # load postgres sales schema + sqlite databases
make test       # run the full test suite
```

## What this provides
- Postgres `sales` schema: 11 normalized tables supporting 6+-table joins.
- SQLite `financials.db` (income statement + stock prices) and `org.db` (headcount).
- Deterministic, reproducible synthetic NVIDIA-themed data (seeded RNG).

## Next plans
2. Mock enterprise APIs · 3. Graph ingestion · 4. deepagents agent · 5. Web app.
```

- [ ] **Step 4: Commit**

```bash
git add backend/README.md
git commit -m "docs(backend): Plan 1 setup and verification README"
```

---

## Self-Review

**Spec coverage (Plan 1 scope):** scaffold (Task 1) ✓ · docker-compose Neo4j+Postgres + Makefile (Task 2) ✓ · 11-table normalized Postgres `sales` enabling 6+-joins (Tasks 3, 4, 7; deep-join test in Task 7) ✓ · SQLite `financials.db` + `org.db` (Tasks 5, 6, 8) ✓ · synthetic NVIDIA-themed data (all generators) ✓ · health checks / startup readiness (compose healthchecks + `make up` wait + conftest skip) ✓. Graph ingestion, APIs, agent, and UI are deliberately deferred to Plans 2–5.

**Placeholder scan:** no TBD/TODO/"handle edge cases"; every code step contains complete runnable code and every run step states the exact command + expected result.

**Type/name consistency:** `build_dimensions()` keys (`regions`, `countries`, `industries`, `segments`, `architectures`, `product_lines`, `fiscal_periods`) are produced in `reference.py` (Task 3) and consumed unchanged by `generate_sales()` (Task 4) and `seed_postgres._DATA_KEY` (Task 7). `generate_sales()` output keys (`products`, `customers`, `sales_orders`, `order_lines`) match `_DATA_KEY` exactly. `generate_financials()` keys (`income_statement`, `stock_prices`) and `generate_org()` keys (`departments`, `locations`, `headcount`) match the inserts in `seed_sqlite.py` (Task 8). Column lists in `_TABLES`/`_insert` match the DDL column names in `postgres_sales.sql` and the SQLite DDL. `settings.postgres_dsn`, `settings.sqlite_dir`, `settings.random_seed` defined in Task 1 and used consistently throughout.
