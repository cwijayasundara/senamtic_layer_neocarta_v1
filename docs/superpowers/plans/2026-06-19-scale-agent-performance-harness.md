# Scale Agent Performance Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repeatable harness that grows the semantic-layer catalog to ~1,000 tables / ~50 APIs and scores the real agent's routing accuracy, answer correctness, and latency so regressions at scale are visible as numbers.

**Architecture:** A deterministic generator emits synthetic distractor "domains" (empty Postgres tables + synthetic OpenAPI specs) around a high-volume answerable core. The existing ingest path introspects them unchanged. A golden NL question set is run through the real agent (`agent.build.ask` + `agent.routing`) by an eval runner that computes routing precision/recall@k, answer-assertion correctness, and per-stage latency, emitting a baseline-vs-scaled scorecard.

**Tech Stack:** Python 3.11+, pydantic-settings, psycopg 3, Faker, FastAPI/OpenAPI dicts, Neo4j, pytest (markers: `postgres`, `neo4j`, `openai`).

## Global Constraints

- Python `>=3.11`; run all commands from `backend/` (package roots: `semantic_layer*`, `data*`).
- Test runner: `cd backend && python -m pytest -v`. Markers gate external deps: `@pytest.mark.postgres`, `@pytest.mark.neo4j`, `@pytest.mark.openai`. Pure tests use no marker.
- Postgres driver is `psycopg` (v3) — `with psycopg.connect(dsn) as conn, conn.cursor() as cur:`.
- Table id format is `table:{source}.{schema}.{name}` (see `graph/schema_ids.table_id`). Core sales table ids look like `table:sales_pg.sales.order_line`.
- Distractor tables are **empty** (metadata only); only the answerable core holds rows. Distractor schemas are namespaced `scale_*` and never touch the `sales` schema.
- All generators are seed-deterministic: same `seed` + size → identical output.
- New config fields must have defaults so non-scale runs are byte-for-byte unaffected.
- Commit after every task with the message shown in its final step.

---

### Task 1: Scale-catalog generator

**Files:**
- Create: `backend/data/generators/scale_catalog.py`
- Test: `backend/tests/test_scale_catalog.py`

**Interfaces:**
- Produces: `generate_scale_catalog(seed: int = 42, n_tables: int = 1000, n_apis: int = 46) -> ScaleCatalog`; dataclasses `ColumnDef(name, type, is_pk=False, ref=None)`, `TableDef(schema, name, columns: list[ColumnDef])`, `ApiField(name, type)`, `ApiDef(source, resource, fields: list[ApiField])`, `ScaleCatalog(tables: list[TableDef], apis: list[ApiDef])`. The `tables` list is topologically ordered: every `ref` target table appears before the table that references it. `ref` is `"schema.table.column"` or `None`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_scale_catalog.py
from data.generators.scale_catalog import generate_scale_catalog, TableDef, ApiDef


def test_generates_requested_table_count():
    cat = generate_scale_catalog(seed=42, n_tables=200, n_apis=10)
    assert len(cat.tables) == 200
    assert len(cat.apis) == 10
    assert all(isinstance(t, TableDef) for t in cat.tables)
    assert all(isinstance(a, ApiDef) for a in cat.apis)


def test_is_deterministic():
    a = generate_scale_catalog(seed=7, n_tables=120, n_apis=8)
    b = generate_scale_catalog(seed=7, n_tables=120, n_apis=8)
    assert [(t.schema, t.name) for t in a.tables] == [(t.schema, t.name) for t in b.tables]


def test_fks_reference_earlier_tables_only():
    cat = generate_scale_catalog(seed=42, n_tables=300, n_apis=5)
    seen = set()
    for t in cat.tables:
        for c in t.columns:
            if c.ref:
                ref_schema, ref_table, _ = c.ref.split(".")
                assert (ref_schema, ref_table) in seen, f"{t.schema}.{t.name} refs unseen {c.ref}"
        seen.add((t.schema, t.name))


def test_includes_near_miss_tables():
    cat = generate_scale_catalog(seed=42, n_tables=300, n_apis=5)
    names = {f"{t.schema}.{t.name}" for t in cat.tables}
    # near-miss tables collide with core keywords (revenue/order/customer)
    assert any("revenue" in n or "order" in n or "customer" in n for n in names)


def test_schemas_are_namespaced():
    cat = generate_scale_catalog(seed=42, n_tables=100, n_apis=5)
    assert all(t.schema.startswith("scale_") for t in cat.tables)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_scale_catalog.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.generators.scale_catalog'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/data/generators/scale_catalog.py
"""Deterministic generator for a large *distractor* catalog: synthetic business
domains of empty tables + synthetic REST APIs that surround the answerable core,
so schema routing must discriminate the real tables from realistic noise."""

import random
from dataclasses import dataclass, field


@dataclass
class ColumnDef:
    name: str
    type: str               # SQL type: INTEGER | TEXT | NUMERIC | DATE | BOOLEAN
    is_pk: bool = False
    ref: str | None = None  # FK target "schema.table.column", or None


@dataclass
class TableDef:
    schema: str             # e.g. "scale_hr"
    name: str               # e.g. "employee"
    columns: list[ColumnDef]


@dataclass
class ApiField:
    name: str
    type: str               # OpenAPI type: integer | string | number


@dataclass
class ApiDef:
    source: str             # e.g. "scale_hr_api"
    resource: str           # e.g. "employees"
    fields: list[ApiField]


@dataclass
class ScaleCatalog:
    tables: list[TableDef] = field(default_factory=list)
    apis: list[ApiDef] = field(default_factory=list)


# Realistic business domains. Each becomes a scale_<domain> schema. The names and
# noun pools are deliberately business-plausible so routing faces real distractors.
_DOMAINS = {
    "hr": ["employee", "department", "payroll", "benefit", "leave_request", "review"],
    "logistics": ["shipment", "warehouse", "route", "carrier", "manifest", "delivery"],
    "marketing": ["campaign", "lead", "channel", "spend", "impression", "conversion"],
    "iot": ["device", "sensor", "reading", "firmware", "alert", "gateway"],
    "supply_chain": ["supplier", "purchase_order", "material", "lot", "receipt", "forecast"],
    "finance_ops": ["invoice", "ledger_entry", "cost_center", "budget", "accrual", "tax_code"],
    "support_ops": ["case", "agent", "queue", "survey", "escalation", "knowledge_article"],
}

# Near-miss nouns collide with answerable-core terms (sales revenue/orders/customers).
_NEAR_MISS = {
    "marketing": [("campaign_revenue", "revenue")],
    "finance_ops": [("quarterly_revenue", "revenue"), ("customer_invoice", "customer")],
    "logistics": [("order_shipment", "order")],
}

_COL_POOL = [
    ("name", "TEXT"), ("status", "TEXT"), ("created_at", "DATE"),
    ("amount", "NUMERIC"), ("quantity", "INTEGER"), ("active", "BOOLEAN"),
    ("region", "TEXT"), ("notes", "TEXT"), ("code", "TEXT"),
]


def _columns_for(rng, schema, table, earlier):
    """Build columns: a PK, 2-4 attributes, and 0-1 FK to an earlier table."""
    cols = [ColumnDef(name=f"{table}_id", type="INTEGER", is_pk=True)]
    for cname, ctype in rng.sample(_COL_POOL, rng.randint(2, 4)):
        cols.append(ColumnDef(name=cname, type=ctype))
    if earlier and rng.random() < 0.6:
        tgt_schema, tgt_table = rng.choice(earlier)
        cols.append(ColumnDef(
            name=f"{tgt_table}_id", type="INTEGER",
            ref=f"{tgt_schema}.{tgt_table}.{tgt_table}_id",
        ))
    return cols


def generate_scale_catalog(seed: int = 42, n_tables: int = 1000, n_apis: int = 46) -> ScaleCatalog:
    rng = random.Random(seed)
    cat = ScaleCatalog()
    earlier: list[tuple[str, str]] = []          # (schema, table) already emitted
    domains = list(_DOMAINS.items())

    # Reserve slots for the fixed set of near-miss tables so they survive the final
    # count (they are appended after the round-robin fill).
    near_miss_total = sum(len(p) for p in _NEAR_MISS.values())
    target_round_robin = max(0, n_tables - near_miss_total)

    # Round-robin tables across domains, suffixing to reach the fill target uniquely.
    i = 0
    while len(cat.tables) < target_round_robin:
        domain, nouns = domains[i % len(domains)]
        schema = f"scale_{domain}"
        noun = nouns[(i // len(domains)) % len(nouns)]
        suffix = i // (len(domains) * len(nouns))
        name = noun if suffix == 0 else f"{noun}_{suffix}"
        cols = _columns_for(rng, schema, name, earlier)
        cat.tables.append(TableDef(schema=schema, name=name, columns=cols))
        earlier.append((schema, name))
        i += 1

    # Append near-miss tables so a few distractor names collide with core keywords.
    # Each references an earlier same-schema table when one exists.
    for domain, pairs in _NEAR_MISS.items():
        schema = f"scale_{domain}"
        prior = [(s, t) for (s, t) in earlier if s == schema]
        for nm_name, _kw in pairs:
            cols = [ColumnDef(name=f"{nm_name}_id", type="INTEGER", is_pk=True),
                    ColumnDef(name="amount", type="NUMERIC"),
                    ColumnDef(name="period", type="TEXT")]
            if prior:
                s, t = prior[0]
                cols.append(ColumnDef(name=f"{t}_id", type="INTEGER", ref=f"{s}.{t}.{t}_id"))
            cat.tables.append(TableDef(schema=schema, name=nm_name, columns=cols))
            earlier.append((schema, nm_name))

    # Safety: never exceed the requested count (near_miss_total < n_tables in practice).
    cat.tables = cat.tables[:n_tables]

    # Synthetic REST APIs: one resource each, drawn from the same domains.
    for j in range(n_apis):
        domain, nouns = domains[j % len(domains)]
        noun = nouns[j % len(nouns)]
        resource = f"{noun}s_{j}"
        fields = [ApiField(name=f"{noun}_id", type="integer"),
                  ApiField(name="name", type="string"),
                  ApiField(name="status", type="string"),
                  ApiField(name="amount", type="number")]
        cat.apis.append(ApiDef(source=f"scale_{domain}_api_{j}", resource=resource, fields=fields))

    return cat
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_scale_catalog.py -v`
Expected: PASS (5 tests). Note: near-miss injection may push count slightly; the final slice `cat.tables[:n_tables]` guarantees exact `n_tables`.

- [ ] **Step 5: Commit**

```bash
git add backend/data/generators/scale_catalog.py backend/tests/test_scale_catalog.py
git commit -m "feat(scale): deterministic distractor catalog generator"
```

---

### Task 2: Config knobs for scale mode

**Files:**
- Modify: `backend/semantic_layer/config.py:44-58` (add fields after the schema-routing block)
- Test: `backend/tests/test_scale_config.py`

**Interfaces:**
- Produces: `settings.scale_mode: bool`, `settings.scale_n_tables: int`, `settings.scale_n_apis: int`, `settings.scale_core_customers: int`, `settings.scale_core_orders: int`, `settings.fake_embeddings: bool`. All read from env via pydantic-settings (uppercase env names, e.g. `SCALE_MODE=true`).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_scale_config.py
from semantic_layer.config import Settings


def test_scale_defaults_are_inert():
    s = Settings()
    assert s.scale_mode is False
    assert s.fake_embeddings is False
    assert s.scale_n_tables == 1000
    assert s.scale_n_apis == 46
    assert s.scale_core_customers == 5000
    assert s.scale_core_orders == 50000


def test_scale_mode_reads_env(monkeypatch):
    monkeypatch.setenv("SCALE_MODE", "true")
    monkeypatch.setenv("SCALE_N_TABLES", "250")
    s = Settings()
    assert s.scale_mode is True
    assert s.scale_n_tables == 250
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_scale_config.py -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'scale_mode'`

- [ ] **Step 3: Write minimal implementation**

In `backend/semantic_layer/config.py`, immediately after the existing line
`schema_routing_max_targets: int = 8` (line 49), insert:

```python

    # --- scale / load-test harness ---
    scale_mode: bool = False
    scale_n_tables: int = 1000
    scale_n_apis: int = 46
    scale_core_customers: int = 5000
    scale_core_orders: int = 50000
    fake_embeddings: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_scale_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/config.py backend/tests/test_scale_config.py
git commit -m "feat(scale): config knobs for scale mode and fake embeddings"
```

---

### Task 3: Distractor seeder + core scale-up + teardown

**Files:**
- Modify: `backend/data/seed_postgres.py:31` (extend `seed(...)` with `n_customers`/`n_orders`)
- Create: `backend/data/seed_scale.py`
- Test: `backend/tests/test_seed_scale.py`

**Interfaces:**
- Consumes: `generate_scale_catalog` (Task 1), `settings` (Task 2), `data.seed_postgres.seed`.
- Produces: `create_distractor_tables(dsn: str, catalog: ScaleCatalog) -> int` (returns table count created); `drop_scale_schemas(dsn: str) -> None`; `seed_scale(dsn=None, seed_value=None) -> dict` (re-seeds core at scale volume **and** creates distractor tables, returns counts). DDL maps `ColumnDef.type` verbatim; PKs become `PRIMARY KEY`, refs become `REFERENCES`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_seed_scale.py
import psycopg
import pytest

from data.generators.scale_catalog import generate_scale_catalog
from data.seed_scale import create_distractor_tables, drop_scale_schemas


@pytest.mark.postgres
def test_create_and_drop_distractor_tables(postgres_dsn):
    cat = generate_scale_catalog(seed=42, n_tables=30, n_apis=3)
    try:
        n = create_distractor_tables(postgres_dsn, cat)
        assert n == 30
        with psycopg.connect(postgres_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema LIKE 'scale_%'"
            )
            assert cur.fetchone()[0] == 30
            # tables are empty (catalog-only)
            t = cat.tables[0]
            cur.execute(f'SELECT count(*) FROM {t.schema}."{t.name}"')
            assert cur.fetchone()[0] == 0
    finally:
        drop_scale_schemas(postgres_dsn)
    with psycopg.connect(postgres_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM information_schema.schemata WHERE schema_name LIKE 'scale_%'"
        )
        assert cur.fetchone()[0] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_seed_scale.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.seed_scale'` (or SKIP if no Postgres — start it with `make up` first).

- [ ] **Step 3a: Extend the core seeder**

In `backend/data/seed_postgres.py`, change the `seed` signature and the
`generate_sales` call. Current (line 31-33):

```python
def seed(dsn: str | None = None, seed_value: int | None = None) -> dict:
    dsn = dsn or settings.postgres_dsn
    data = generate_sales(seed=seed_value if seed_value is not None else settings.random_seed)
```

Replace with:

```python
def seed(dsn: str | None = None, seed_value: int | None = None,
         n_customers: int | None = None, n_orders: int | None = None) -> dict:
    dsn = dsn or settings.postgres_dsn
    kwargs = {"seed": seed_value if seed_value is not None else settings.random_seed}
    if n_customers is not None:
        kwargs["n_customers"] = n_customers
    if n_orders is not None:
        kwargs["n_orders"] = n_orders
    data = generate_sales(**kwargs)
```

- [ ] **Step 3b: Write the distractor seeder**

```python
# backend/data/seed_scale.py
"""Materialize the distractor catalog as EMPTY Postgres tables (catalog-only) and
re-seed the answerable core at scale volume. Distractor schemas are namespaced
`scale_*` and fully removable via drop_scale_schemas()."""

import psycopg

from data.generators.scale_catalog import ScaleCatalog, TableDef, generate_scale_catalog
from data.seed_postgres import seed as seed_core
from semantic_layer.config import settings


def _create_table_sql(t: TableDef) -> str:
    cols = []
    for c in t.columns:
        parts = [f'"{c.name}"', c.type]
        if c.is_pk:
            parts.append("PRIMARY KEY")
        if c.ref:
            ref_schema, ref_table, ref_col = c.ref.split(".")
            parts.append(f'REFERENCES {ref_schema}."{ref_table}" ("{ref_col}")')
        cols.append(" ".join(parts))
    body = ",\n  ".join(cols)
    return f'CREATE TABLE IF NOT EXISTS {t.schema}."{t.name}" (\n  {body}\n);'


def create_distractor_tables(dsn: str, catalog: ScaleCatalog) -> int:
    """Create every catalog table (empty) in FK-valid order. Idempotent."""
    schemas = sorted({t.schema for t in catalog.tables})
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        for s in schemas:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {s}")
        for t in catalog.tables:          # generator guarantees topological order
            cur.execute(_create_table_sql(t))
        conn.commit()
    return len(catalog.tables)


def drop_scale_schemas(dsn: str) -> None:
    """Drop all scale_* schemas, restoring the baseline DB."""
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT schema_name FROM information_schema.schemata WHERE schema_name LIKE 'scale_%'"
        )
        for (name,) in cur.fetchall():
            cur.execute(f"DROP SCHEMA {name} CASCADE")
        conn.commit()


def seed_scale(dsn: str | None = None, seed_value: int | None = None) -> dict:
    dsn = dsn or settings.postgres_dsn
    core = seed_core(
        dsn=dsn, seed_value=seed_value,
        n_customers=settings.scale_core_customers,
        n_orders=settings.scale_core_orders,
    )
    catalog = generate_scale_catalog(
        seed=settings.random_seed,
        n_tables=settings.scale_n_tables,
        n_apis=settings.scale_n_apis,
    )
    n = create_distractor_tables(dsn, catalog)
    return {"core_tables": len(core), "distractor_tables": n}


if __name__ == "__main__":
    print(seed_scale())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && make up && python -m pytest tests/test_seed_scale.py -v`
Expected: PASS (1 test). If Postgres is unreachable the test SKIPs — start it with `make up`.

- [ ] **Step 5: Commit**

```bash
git add backend/data/seed_scale.py backend/data/seed_postgres.py backend/tests/test_seed_scale.py
git commit -m "feat(scale): empty distractor-table seeder + core scale-up + teardown"
```

---

### Task 4: Synthetic OpenAPI specs for distractor APIs

**Files:**
- Create: `backend/semantic_layer/ingest/synthetic_api.py`
- Test: `backend/tests/test_synthetic_api.py`

**Interfaces:**
- Consumes: `ApiDef` (Task 1), `extract_openapi`/`SchemaBundle` (`ingest.api_extractor`).
- Produces: `build_openapi_spec(api: ApiDef) -> dict` (OpenAPI 3.1 dict compatible with `extract_openapi`); `extract_synthetic_apis(catalog: ScaleCatalog) -> list[SchemaBundle]`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_synthetic_api.py
from data.generators.scale_catalog import generate_scale_catalog
from semantic_layer.ingest.api_extractor import extract_openapi
from semantic_layer.ingest.synthetic_api import build_openapi_spec, extract_synthetic_apis


def test_spec_parses_into_virtual_table():
    cat = generate_scale_catalog(seed=42, n_tables=10, n_apis=2)
    api = cat.apis[0]
    spec = build_openapi_spec(api)
    bundle = extract_openapi(spec, source=api.source)
    table_names = {t.name for t in bundle.tables}
    assert f"GET /{api.resource}" in table_names
    col_names = {c.name for c in bundle.columns}
    assert {f.name for f in api.fields} <= col_names


def test_extract_synthetic_apis_one_bundle_per_api():
    cat = generate_scale_catalog(seed=42, n_tables=10, n_apis=5)
    bundles = extract_synthetic_apis(cat)
    assert len(bundles) == 5
    assert all(b.tables for b in bundles)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_synthetic_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'semantic_layer.ingest.synthetic_api'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/semantic_layer/ingest/synthetic_api.py
"""Turn synthetic ApiDef entries into OpenAPI 3.1 spec dicts that extract_openapi
can introspect, so distractor REST endpoints land in the graph on the real path."""

from data.generators.scale_catalog import ApiDef, ScaleCatalog
from semantic_layer.ingest.api_extractor import extract_openapi
from semantic_layer.ingest.sql_extractor import SchemaBundle


def build_openapi_spec(api: ApiDef) -> dict:
    model = api.resource.replace("_", " ").title().replace(" ", "")
    return {
        "openapi": "3.1.0",
        "info": {"title": api.source, "version": "1.0.0"},
        "paths": {
            f"/{api.resource}": {
                "get": {
                    "summary": f"List {api.resource}",
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {"$ref": f"#/components/schemas/{model}"},
                                    }
                                }
                            }
                        }
                    },
                }
            }
        },
        "components": {
            "schemas": {
                model: {
                    "type": "object",
                    "properties": {f.name: {"type": f.type} for f in api.fields},
                }
            }
        },
    }


def extract_synthetic_apis(catalog: ScaleCatalog) -> list[SchemaBundle]:
    return [extract_openapi(build_openapi_spec(a), source=a.source) for a in catalog.apis]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_synthetic_api.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/ingest/synthetic_api.py backend/tests/test_synthetic_api.py
git commit -m "feat(scale): synthetic OpenAPI specs for distractor APIs"
```

---

### Task 5: Pipeline scale branch

**Files:**
- Modify: `backend/semantic_layer/ingest/pipeline.py:33-49` (add distractor + synthetic-API bundles when `scale_mode`)
- Test: `backend/tests/test_pipeline_scale.py`

**Interfaces:**
- Consumes: `generate_scale_catalog` (Task 1), `extract_synthetic_apis` (Task 4), `extract_postgres`, `settings.scale_mode/scale_n_tables/scale_n_apis`.
- Produces: a private helper `_scale_bundles() -> list[SchemaBundle]` and an extended `run_ingest` that appends scale bundles when `settings.scale_mode`. `counts["scale_sources"]` records how many scale bundles were loaded.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_pipeline_scale.py
from semantic_layer.ingest import pipeline


def test_scale_bundles_empty_when_disabled(monkeypatch):
    monkeypatch.setattr(pipeline.settings, "scale_mode", False)
    assert pipeline._scale_bundles() == []


def test_scale_bundles_built_when_enabled(monkeypatch):
    monkeypatch.setattr(pipeline.settings, "scale_mode", True)
    monkeypatch.setattr(pipeline.settings, "scale_n_tables", 12)
    monkeypatch.setattr(pipeline.settings, "scale_n_apis", 3)
    # extract_postgres needs a DB; stub it to isolate the assembly logic.
    captured = {}

    def fake_extract_postgres(dsn, source="sales_pg", schema_name="sales"):
        from semantic_layer.ingest.sql_extractor import SchemaBundle
        captured.setdefault("schemas", []).append(schema_name)
        return SchemaBundle()

    monkeypatch.setattr(pipeline, "extract_postgres", fake_extract_postgres)
    bundles = pipeline._scale_bundles()
    # one bundle per distractor schema (scale_*) + one per synthetic API (3)
    assert len(captured["schemas"]) >= 1
    assert all(s.startswith("scale_") for s in captured["schemas"])
    assert len(bundles) == len(captured["schemas"]) + 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_pipeline_scale.py -v`
Expected: FAIL with `AttributeError: module 'semantic_layer.ingest.pipeline' has no attribute '_scale_bundles'`

- [ ] **Step 3: Write minimal implementation**

In `backend/semantic_layer/ingest/pipeline.py`, add imports near the existing
ingest imports (after line 18):

```python
from semantic_layer.ingest.synthetic_api import extract_synthetic_apis
from data.generators.scale_catalog import generate_scale_catalog
```

Add the helper above `run_ingest` (after `_api_spec_getter`, line 30):

```python
def _scale_bundles() -> list:
    """Distractor SchemaBundles: one per scale_* schema (empty Postgres tables,
    introspected live) plus one per synthetic REST API. Empty when scale_mode off."""
    if not settings.scale_mode:
        return []
    catalog = generate_scale_catalog(
        seed=settings.random_seed,
        n_tables=settings.scale_n_tables,
        n_apis=settings.scale_n_apis,
    )
    schemas = sorted({t.schema for t in catalog.tables})
    bundles = [
        extract_postgres(settings.postgres_dsn, source="scale", schema_name=s)
        for s in schemas
    ]
    bundles += extract_synthetic_apis(catalog)
    return bundles
```

In `run_ingest`, after the line `bundles += extract_all_apis(_api_spec_getter(), settings.api_source_list)` (line 46), insert:

```python
        scale = _scale_bundles()
        bundles += scale
        counts["scale_sources"] = len(scale)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_pipeline_scale.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/ingest/pipeline.py backend/tests/test_pipeline_scale.py
git commit -m "feat(scale): ingest distractor schemas + synthetic APIs under scale_mode"
```

---

### Task 6: Fake embeddings (cost control)

**Files:**
- Modify: `backend/semantic_layer/ingest/embeddings.py` (branch both embed fns on `settings.fake_embeddings`)
- Test: `backend/tests/test_fake_embeddings.py`

**Interfaces:**
- Consumes: `settings.fake_embeddings`, `settings.embedding_dimensions`.
- Produces: `fake_vector(text: str, dim: int) -> list[float]` (deterministic, unit-norm-ish). `embed_chunks` writes fake vectors when the flag is on; `embed_metadata_nodes` becomes a no-op when the flag is on (routing uses keyword search, not metadata vectors — see `agent/routing.py`).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_fake_embeddings.py
from semantic_layer.ingest.embeddings import fake_vector


def test_fake_vector_is_deterministic_and_sized():
    a = fake_vector("total revenue by region", 1536)
    b = fake_vector("total revenue by region", 1536)
    assert a == b
    assert len(a) == 1536
    assert all(isinstance(x, float) for x in a)


def test_fake_vector_differs_by_text():
    assert fake_vector("alpha", 64) != fake_vector("beta", 64)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_fake_embeddings.py -v`
Expected: FAIL with `ImportError: cannot import name 'fake_vector'`

- [ ] **Step 3: Write minimal implementation**

In `backend/semantic_layer/ingest/embeddings.py`, add at the top after the
existing imports:

```python
import hashlib
import struct
```

Add the helper near the top of the module (after the imports, before
`embed_chunks`):

```python
def fake_vector(text: str, dim: int) -> list[float]:
    """Deterministic pseudo-embedding from a text hash — no OpenAI call. For scale
    runs where exact semantic quality is not under test (routing is keyword-based)."""
    out: list[float] = []
    i = 0
    while len(out) < dim:
        digest = hashlib.sha256(f"{text}:{i}".encode()).digest()
        for j in range(0, len(digest), 4):
            if len(out) >= dim:
                break
            (val,) = struct.unpack("I", digest[j:j + 4])
            out.append((val / 0xFFFFFFFF) * 2.0 - 1.0)   # in [-1, 1]
        i += 1
    return out
```

At the start of `embed_chunks` (right after the docstring, before
`client = get_openai_client()`), insert the fake branch:

```python
    if settings.fake_embeddings:
        with driver.session(database=settings.neo4j_database) as session:
            rows = session.run(
                "MATCH (c:Chunk) WHERE c.embedding IS NULL RETURN c.id AS id, c.text AS text"
            ).data()
            session.run(
                """
                UNWIND $rows AS row
                MATCH (c:Chunk {id: row.id})
                CALL db.create.setNodeVectorProperty(c, 'embedding', row.vec)
                """,
                rows=[{"id": r["id"], "vec": fake_vector(r["text"] or "", settings.embedding_dimensions)}
                      for r in rows],
            )
        _ensure_chunk_vector_index(driver)
        return
```

At the start of `embed_metadata_nodes` (right after the docstring), insert:

```python
    if settings.fake_embeddings:
        return  # routing uses keyword catalog search; skip costly metadata embeds
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_fake_embeddings.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/ingest/embeddings.py backend/tests/test_fake_embeddings.py
git commit -m "feat(scale): deterministic fake embeddings for cost-free scale ingest"
```

---

### Task 7: Golden question set + loader

**Files:**
- Create: `backend/eval/__init__.py` (empty)
- Create: `backend/eval/golden_questions.json`
- Create: `backend/eval/golden.py`
- Test: `backend/tests/test_golden.py`

**Interfaces:**
- Produces: `GoldenQuestion` (dataclass: `id, question, expected_tables: list[str], answer_check: dict, category: str`); `load_golden(path: str | None = None) -> list[GoldenQuestion]`. `expected_tables` use full table-id form (`table:sales_pg.sales.order_line`). `answer_check` is `{"type": "contains", "values": [...]}` or `{"type": "numeric", "value": <num>, "tol": <num>}`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_golden.py
from eval.golden import load_golden, GoldenQuestion

VALID_CATEGORIES = {
    "single-table-agg", "multi-table-join", "cross-source-bridge",
    "document-rag", "period-filter", "near-miss-routing",
}


def test_golden_loads_and_is_well_formed():
    qs = load_golden()
    assert len(qs) >= 20
    assert all(isinstance(q, GoldenQuestion) for q in qs)


def test_every_question_has_required_fields():
    for q in load_golden():
        assert q.id and q.question
        assert q.category in VALID_CATEGORIES
        assert isinstance(q.expected_tables, list)
        assert q.answer_check.get("type") in {"contains", "numeric"}
        if q.category != "near-miss-routing":
            assert q.expected_tables, f"{q.id} needs expected_tables"


def test_table_ids_are_fully_qualified():
    for q in load_golden():
        for t in q.expected_tables:
            assert t.startswith("table:"), f"{q.id}: {t} not a table id"


def test_ids_are_unique():
    ids = [q.id for q in load_golden()]
    assert len(ids) == len(set(ids))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_golden.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.golden'`

- [ ] **Step 3a: Write the loader**

```python
# backend/eval/__init__.py
```

```python
# backend/eval/golden.py
"""Load the golden NL-question set used to score the agent at scale."""

import json
from dataclasses import dataclass
from pathlib import Path

_DEFAULT = Path(__file__).with_name("golden_questions.json")


@dataclass
class GoldenQuestion:
    id: str
    question: str
    expected_tables: list[str]
    answer_check: dict
    category: str


def load_golden(path: str | None = None) -> list[GoldenQuestion]:
    data = json.loads(Path(path or _DEFAULT).read_text())
    return [GoldenQuestion(**row) for row in data]
```

- [ ] **Step 3b: Write the golden set**

Create `backend/eval/golden_questions.json`. The core sales schema (table ids
`table:sales_pg.sales.<t>`) has: `region, country, industry, customer, segment,
architecture, product_line, product, fiscal_period, sales_order, order_line`. The
financials SQLite source is `table:financials.main.<t>` (`income_statement,
stock_price`); org is `table:org.main.<t>` (`department, location, headcount`).
Cross-source CRM API endpoints are `table:crm.api.GET /accounts` etc. Provide at
least 20 questions covering all six categories:

```json
[
  {"id": "agg-total-revenue", "question": "What was the total order-line revenue across all time?", "expected_tables": ["table:sales_pg.sales.order_line"], "answer_check": {"type": "contains", "values": ["revenue"]}, "category": "single-table-agg"},
  {"id": "agg-customer-count", "question": "How many customers are there in total?", "expected_tables": ["table:sales_pg.sales.customer"], "answer_check": {"type": "numeric", "value": 5000, "tol": 0}, "category": "single-table-agg"},
  {"id": "agg-product-count", "question": "How many products do we sell?", "expected_tables": ["table:sales_pg.sales.product"], "answer_check": {"type": "numeric", "value": 20, "tol": 0}, "category": "single-table-agg"},
  {"id": "agg-order-count", "question": "How many sales orders were placed?", "expected_tables": ["table:sales_pg.sales.sales_order"], "answer_check": {"type": "numeric", "value": 50000, "tol": 0}, "category": "single-table-agg"},
  {"id": "join-revenue-by-region", "question": "What is total revenue by region?", "expected_tables": ["table:sales_pg.sales.order_line", "table:sales_pg.sales.sales_order", "table:sales_pg.sales.customer", "table:sales_pg.sales.country", "table:sales_pg.sales.region"], "answer_check": {"type": "contains", "values": ["region"]}, "category": "multi-table-join"},
  {"id": "join-revenue-by-industry", "question": "Which industry generated the most revenue?", "expected_tables": ["table:sales_pg.sales.order_line", "table:sales_pg.sales.sales_order", "table:sales_pg.sales.customer", "table:sales_pg.sales.industry"], "answer_check": {"type": "contains", "values": ["industry"]}, "category": "multi-table-join"},
  {"id": "join-revenue-by-segment", "question": "Show revenue by business segment.", "expected_tables": ["table:sales_pg.sales.order_line", "table:sales_pg.sales.product", "table:sales_pg.sales.product_line", "table:sales_pg.sales.segment"], "answer_check": {"type": "contains", "values": ["segment"]}, "category": "multi-table-join"},
  {"id": "join-revenue-by-product-line", "question": "What is total revenue per product line?", "expected_tables": ["table:sales_pg.sales.order_line", "table:sales_pg.sales.product", "table:sales_pg.sales.product_line"], "answer_check": {"type": "contains", "values": ["product"]}, "category": "multi-table-join"},
  {"id": "join-top-customer", "question": "Which customer has the highest total order value?", "expected_tables": ["table:sales_pg.sales.order_line", "table:sales_pg.sales.sales_order", "table:sales_pg.sales.customer"], "answer_check": {"type": "contains", "values": ["customer"]}, "category": "multi-table-join"},
  {"id": "period-revenue-fy2025", "question": "What was total revenue in fiscal year 2025?", "expected_tables": ["table:sales_pg.sales.order_line", "table:sales_pg.sales.sales_order", "table:sales_pg.sales.fiscal_period"], "answer_check": {"type": "contains", "values": ["2025"]}, "category": "period-filter"},
  {"id": "period-orders-by-quarter", "question": "How many orders were placed each fiscal quarter?", "expected_tables": ["table:sales_pg.sales.sales_order", "table:sales_pg.sales.fiscal_period"], "answer_check": {"type": "contains", "values": ["quarter"]}, "category": "period-filter"},
  {"id": "period-revenue-trend", "question": "Show the revenue trend by fiscal period.", "expected_tables": ["table:sales_pg.sales.order_line", "table:sales_pg.sales.sales_order", "table:sales_pg.sales.fiscal_period"], "answer_check": {"type": "contains", "values": ["period"]}, "category": "period-filter"},
  {"id": "fin-net-income", "question": "What was reported net income by fiscal year?", "expected_tables": ["table:financials.main.income_statement"], "answer_check": {"type": "contains", "values": ["income"]}, "category": "single-table-agg"},
  {"id": "fin-stock-high", "question": "What was the highest closing stock price?", "expected_tables": ["table:financials.main.stock_price"], "answer_check": {"type": "contains", "values": ["price"]}, "category": "single-table-agg"},
  {"id": "org-headcount-by-dept", "question": "What is headcount by department?", "expected_tables": ["table:org.main.headcount", "table:org.main.department"], "answer_check": {"type": "contains", "values": ["department"]}, "category": "multi-table-join"},
  {"id": "org-locations", "question": "How many office locations are there?", "expected_tables": ["table:org.main.location"], "answer_check": {"type": "contains", "values": ["location"]}, "category": "single-table-agg"},
  {"id": "bridge-open-opps-by-account", "question": "Which accounts have the most open opportunities?", "expected_tables": ["table:crm.api.GET /opportunities", "table:crm.api.GET /accounts"], "answer_check": {"type": "contains", "values": ["account"]}, "category": "cross-source-bridge"},
  {"id": "bridge-revenue-vs-tickets", "question": "For our top-revenue customers, how many support tickets do they have?", "expected_tables": ["table:sales_pg.sales.order_line", "table:sales_pg.sales.customer", "table:itsm.api.GET /tickets"], "answer_check": {"type": "contains", "values": ["ticket"]}, "category": "cross-source-bridge"},
  {"id": "bridge-gpu-usage-by-account", "question": "What is total GPU usage by account?", "expected_tables": ["table:dgx.api.GET /usage", "table:crm.api.GET /accounts"], "answer_check": {"type": "contains", "values": ["usage"]}, "category": "cross-source-bridge"},
  {"id": "doc-revenue-drivers", "question": "According to the annual report, what drove revenue growth?", "expected_tables": [], "answer_check": {"type": "contains", "values": ["revenue"]}, "category": "document-rag"},
  {"id": "doc-risk-factors", "question": "What risk factors does the annual report highlight?", "expected_tables": [], "answer_check": {"type": "contains", "values": ["risk"]}, "category": "document-rag"},
  {"id": "nearmiss-sales-revenue", "question": "What is our total sales revenue?", "expected_tables": ["table:sales_pg.sales.order_line"], "answer_check": {"type": "contains", "values": ["revenue"]}, "category": "near-miss-routing"},
  {"id": "nearmiss-customer-orders", "question": "How many orders has each customer placed?", "expected_tables": ["table:sales_pg.sales.sales_order", "table:sales_pg.sales.customer"], "answer_check": {"type": "contains", "values": ["customer"]}, "category": "near-miss-routing"}
]
```

(Note: the `near-miss-routing` questions deliberately collide with distractor tables
`scale_marketing.campaign_revenue`, `scale_finance_ops.quarterly_revenue`,
`scale_logistics.order_shipment` from Task 1 — they pass only if routing selects the
core `sales` tables, not the distractors.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_golden.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/eval/__init__.py backend/eval/golden.py backend/eval/golden_questions.json backend/tests/test_golden.py
git commit -m "feat(eval): golden NL question set + loader"
```

---

### Task 8: Scorer (pure metric functions)

**Files:**
- Create: `backend/eval/scorer.py`
- Test: `backend/tests/test_scorer.py`

**Interfaces:**
- Consumes: nothing external (pure).
- Produces: `routing_scores(routed: list[str], expected: list[str]) -> dict` returning `{"precision": float, "recall": float, "hit": bool}` (empty `expected` → precision/recall 1.0, hit True — used by near-miss/doc questions where table set is not asserted); `check_answer(answer: str, check: dict) -> bool` handling `contains` and `numeric`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_scorer.py
from eval.scorer import routing_scores, check_answer


def test_routing_perfect():
    s = routing_scores(["table:a", "table:b"], ["table:a", "table:b"])
    assert s["precision"] == 1.0 and s["recall"] == 1.0 and s["hit"] is True


def test_routing_partial():
    s = routing_scores(["table:a", "table:x"], ["table:a", "table:b"])
    assert s["precision"] == 0.5
    assert s["recall"] == 0.5
    assert s["hit"] is False


def test_routing_empty_expected_is_pass():
    s = routing_scores(["table:a"], [])
    assert s["precision"] == 1.0 and s["recall"] == 1.0 and s["hit"] is True


def test_check_answer_contains():
    assert check_answer("Total revenue by REGION was $5M", {"type": "contains", "values": ["region", "revenue"]})
    assert not check_answer("no match here", {"type": "contains", "values": ["region"]})


def test_check_answer_numeric():
    assert check_answer("There are 5000 customers.", {"type": "numeric", "value": 5000, "tol": 0})
    assert check_answer("about 4998 rows", {"type": "numeric", "value": 5000, "tol": 5})
    assert not check_answer("about 4000 rows", {"type": "numeric", "value": 5000, "tol": 5})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.scorer'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/eval/scorer.py
"""Pure scoring functions: routing precision/recall and answer-assertion checks."""

import re


def routing_scores(routed: list[str], expected: list[str]) -> dict:
    """Precision/recall of routed table ids vs expected. Empty `expected` means the
    question does not assert a table set (doc-RAG / near-miss) -> treated as a pass."""
    if not expected:
        return {"precision": 1.0, "recall": 1.0, "hit": True}
    routed_set, expected_set = set(routed), set(expected)
    inter = routed_set & expected_set
    precision = len(inter) / len(routed_set) if routed_set else 0.0
    recall = len(inter) / len(expected_set)
    return {"precision": precision, "recall": recall, "hit": expected_set <= routed_set}


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def check_answer(answer: str, check: dict) -> bool:
    kind = check.get("type")
    if kind == "contains":
        low = answer.lower()
        return all(v.lower() in low for v in check.get("values", []))
    if kind == "numeric":
        target, tol = float(check["value"]), float(check.get("tol", 0))
        for m in _NUM.findall(answer):
            try:
                if abs(float(m.replace(",", "")) - target) <= tol:
                    return True
            except ValueError:
                continue
        return False
    return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_scorer.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/eval/scorer.py backend/tests/test_scorer.py
git commit -m "feat(eval): pure routing + answer scoring functions"
```

---

### Task 9: Eval runner

**Files:**
- Create: `backend/eval/run_eval.py`
- Test: `backend/tests/test_run_eval.py`

**Interfaces:**
- Consumes: `load_golden` (Task 7), `routing_scores`/`check_answer` (Task 8), `agent.routing.route_tables`, `agent.build.ask`.
- Produces: `evaluate(route_fn, answer_fn, questions, timer=...) -> dict` (dependency-injected for testing) returning a scorecard `{"questions": [...per-q...], "summary": {...}}`; `main(argv=None)` CLI that wires the real `route_tables`/`ask`, writes `scorecard.json`, and prints a summary table. Per-question record: `{id, category, routing: {...}, answer_ok: bool, latency_ms: {route, answer, total}}`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_run_eval.py
from eval.golden import GoldenQuestion
from eval.run_eval import evaluate


def _q(id, expected, check, cat="single-table-agg"):
    return GoldenQuestion(id=id, question="q?", expected_tables=expected,
                          answer_check=check, category=cat)


def test_evaluate_aggregates_metrics():
    questions = [
        _q("a", ["table:x"], {"type": "contains", "values": ["ok"]}),
        _q("b", ["table:y"], {"type": "contains", "values": ["zzz"]}),
    ]
    route_fn = lambda q: {"a": ["table:x"], "b": ["table:w"]}[q.id]
    answer_fn = lambda q: {"a": "ok answer", "b": "wrong"}[q.id]
    clock = iter([0.0, 0.1, 0.2, 1.0, 1.1, 1.2])  # 3 timer() calls per question
    card = evaluate(route_fn, answer_fn, questions, timer=lambda: next(clock))

    assert card["summary"]["n"] == 2
    assert card["summary"]["routing_hit_rate"] == 0.5   # only "a" hits
    assert card["summary"]["answer_accuracy"] == 0.5    # only "a" passes
    by_id = {r["id"]: r for r in card["questions"]}
    assert by_id["a"]["answer_ok"] is True
    assert by_id["b"]["routing"]["hit"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_run_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.run_eval'`

- [ ] **Step 3: Write the implementation**

```python
# backend/eval/run_eval.py
"""Run the golden set through the real agent and emit a routing/answer/latency
scorecard. Routing and answering are injected so the aggregation is unit-testable
without an LLM or a live graph."""

import argparse
import json
import time
from pathlib import Path

from eval.golden import GoldenQuestion, load_golden
from eval.scorer import check_answer, routing_scores


def evaluate(route_fn, answer_fn, questions: list[GoldenQuestion], timer=time.perf_counter) -> dict:
    """route_fn(q: GoldenQuestion) -> list[str] table ids;
    answer_fn(q: GoldenQuestion) -> str answer. Both are injected so the runner is
    testable without an LLM or live graph (main() wires the real agent)."""
    records, latencies = [], []
    hits = answers_ok = 0
    prec_sum = rec_sum = 0.0
    for q in questions:
        t0 = timer()
        routed = route_fn(q)
        t1 = timer()
        answer = answer_fn(q)
        t2 = timer()
        r = routing_scores(routed, q.expected_tables)
        ok = check_answer(answer, q.answer_check)
        hits += int(r["hit"])
        answers_ok += int(ok)
        prec_sum += r["precision"]
        rec_sum += r["recall"]
        lat = {"route": round((t1 - t0) * 1000, 1),
               "answer": round((t2 - t1) * 1000, 1),
               "total": round((t2 - t0) * 1000, 1)}
        latencies.append(lat["total"])
        records.append({"id": q.id, "category": q.category, "routing": r,
                        "routed": routed, "answer_ok": ok, "latency_ms": lat})
    n = len(questions) or 1
    latencies.sort()
    summary = {
        "n": len(questions),
        "routing_hit_rate": round(hits / n, 3),
        "routing_precision": round(prec_sum / n, 3),
        "routing_recall": round(rec_sum / n, 3),
        "answer_accuracy": round(answers_ok / n, 3),
        "latency_p50_ms": latencies[len(latencies) // 2] if latencies else 0.0,
        "latency_p95_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0.0,
    }
    return {"questions": records, "summary": summary}


def _print_summary(card: dict) -> None:
    s = card["summary"]
    print("\n=== Scale Agent Scorecard ===")
    print(f"questions:          {s['n']}")
    print(f"routing hit-rate:   {s['routing_hit_rate']:.1%}")
    print(f"routing precision:  {s['routing_precision']:.3f}")
    print(f"routing recall:     {s['routing_recall']:.3f}")
    print(f"answer accuracy:    {s['answer_accuracy']:.1%}")
    print(f"latency p50/p95 ms: {s['latency_p50_ms']} / {s['latency_p95_ms']}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Score the agent over the golden set.")
    ap.add_argument("--out", default="scorecard.json")
    ap.add_argument("--routing-only", action="store_true",
                    help="skip full answer generation (cheaper; routing metrics only)")
    args = ap.parse_args(argv)

    from semantic_layer.agent.routing import route_tables
    from semantic_layer.config import settings

    def route_fn(q: GoldenQuestion):
        return route_tables(q.question,
                            k_ret=settings.schema_routing_k_ret,
                            k_rank=settings.schema_routing_k_rank)

    if args.routing_only:
        def answer_fn(q: GoldenQuestion):
            return ""
    else:
        from semantic_layer.agent.build import ask

        def answer_fn(q: GoldenQuestion):
            return ask(q.question)

    questions = load_golden()
    card = evaluate(route_fn, answer_fn, questions)
    card["summary"]["embedding_mode"] = "fake" if settings.fake_embeddings else "openai"
    card["summary"]["routing_only"] = args.routing_only
    Path(args.out).write_text(json.dumps(card, indent=2))
    _print_summary(card)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_run_eval.py -v`
Expected: PASS (1 test). `route_fn`/`answer_fn` take the `GoldenQuestion` and the
`clock` yields 3 values per question (route start, answer start, end).

- [ ] **Step 5: Commit**

```bash
git add backend/eval/run_eval.py backend/tests/test_run_eval.py
git commit -m "feat(eval): scorecard runner (routing + answer + latency)"
```

---

### Task 10: Makefile wiring

**Files:**
- Modify: `backend/../Makefile` (repo-root `Makefile`) — add scale + eval targets and `.PHONY`.
- Test: manual smoke (documented below) — no unit test; Make recipes are thin wrappers over already-tested modules.

**Interfaces:**
- Consumes: `data.seed_scale`, `semantic_layer.ingest.pipeline`, `eval.run_eval`, `data.seed_scale.drop_scale_schemas`.
- Produces: targets `scale-seed`, `scale-ingest`, `eval`, `eval-baseline`, `scale-teardown`.

- [ ] **Step 1: Add the targets**

In the repo-root `Makefile`, add `scale-seed scale-ingest eval eval-baseline scale-teardown`
to the `.PHONY` line, then append:

```makefile
# --- scale / agent-performance harness ---
# Seed the answerable core at scale volume + create empty distractor tables.
scale-seed:
	cd backend && SCALE_MODE=true python -m data.seed_scale

# Ingest in scale mode (distractor schemas + synthetic APIs), routing on, fake embeds.
scale-ingest:
	cd backend && SCALE_MODE=true SCHEMA_ROUTING_ENABLED=true FAKE_EMBEDDINGS=true \
		python -m semantic_layer.ingest.pipeline

# Score the agent over the golden set against the current (scaled) graph.
eval:
	cd backend && SCHEMA_ROUTING_ENABLED=true python -m eval.run_eval --out scorecard.json

# Baseline score against the default small catalog (routing off) for comparison.
eval-baseline:
	cd backend && python -m eval.run_eval --out scorecard-baseline.json

# Drop all scale_* schemas, restoring the baseline DB.
scale-teardown:
	cd backend && python -c "from data.seed_scale import drop_scale_schemas; from semantic_layer.config import settings; drop_scale_schemas(settings.postgres_dsn); print('scale schemas dropped')"
```

- [ ] **Step 2: Smoke-test the wiring with a tiny catalog**

Run (requires `make up` Postgres + Neo4j running):

```bash
cd backend && SCALE_MODE=true SCALE_N_TABLES=50 SCALE_N_APIS=5 python -m data.seed_scale
cd backend && SCALE_MODE=true SCALE_N_TABLES=50 SCALE_N_APIS=5 SCHEMA_ROUTING_ENABLED=true FAKE_EMBEDDINGS=true python -m semantic_layer.ingest.pipeline
```

Expected: `seed_scale` prints `{'core_tables': ..., 'distractor_tables': 50}`;
`run_ingest` prints a counts dict including `'scale_sources': <>=~12>` and completes
without OpenAI calls (fake embeddings).

- [ ] **Step 3: Smoke-test the scorer (routing-only, cheap)**

```bash
cd backend && SCHEMA_ROUTING_ENABLED=true python -m eval.run_eval --routing-only --out /tmp/scorecard.json
```

Expected: prints the scorecard summary; `routing hit-rate` is a real fraction;
`/tmp/scorecard.json` exists with per-question records.

- [ ] **Step 4: Tear down**

```bash
cd backend && python -c "from data.seed_scale import drop_scale_schemas; from semantic_layer.config import settings; drop_scale_schemas(settings.postgres_dsn)"
```

Expected: no error; `scale_*` schemas removed.

- [ ] **Step 5: Commit**

```bash
git add Makefile
git commit -m "feat(scale): make targets for scale seed/ingest/eval/teardown"
```

---

### Task 11 (optional): Load / concurrency harness

**Files:**
- Create: `backend/eval/load_test.py`
- Test: `backend/tests/test_load_test.py`

**Interfaces:**
- Consumes: `load_golden` (Task 7).
- Produces: `run_load(send_fn, questions, concurrency: int = 8, rounds: int = 1, timer=...) -> dict` returning `{"n": int, "throughput_qps": float, "latency_p50_ms": float, "latency_p95_ms": float, "errors": int}`. `send_fn(question_text) -> None` is injected (production wires it to an HTTP POST to `/chat`). Only build this if concurrency numbers are needed now; otherwise defer.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_load_test.py
from eval.golden import GoldenQuestion
from eval.load_test import run_load


def test_run_load_counts_and_errors():
    qs = [GoldenQuestion(id=str(i), question="q", expected_tables=[],
                         answer_check={"type": "contains", "values": []},
                         category="single-table-agg") for i in range(4)]
    calls = {"n": 0}

    def send_fn(_q):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")

    clock = iter([float(i) for i in range(100)])
    res = run_load(send_fn, qs, concurrency=2, rounds=1, timer=lambda: next(clock))
    assert res["n"] == 4
    assert res["errors"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_load_test.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.load_test'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/eval/load_test.py
"""Concurrency harness: fire golden questions at an injected send_fn and report
throughput + tail latency. send_fn is wired to POST /chat in production."""

import time
from concurrent.futures import ThreadPoolExecutor

from eval.golden import GoldenQuestion, load_golden


def run_load(send_fn, questions: list[GoldenQuestion], concurrency: int = 8,
             rounds: int = 1, timer=time.perf_counter) -> dict:
    jobs = [q.question for _ in range(rounds) for q in questions]
    latencies, errors = [], 0

    def one(text: str):
        t0 = timer()
        try:
            send_fn(text)
            return (timer() - t0) * 1000, None
        except Exception as exc:  # noqa: BLE001 — load harness records, never raises
            return (timer() - t0) * 1000, exc

    start = timer()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for ms, err in pool.map(one, jobs):
            latencies.append(ms)
            if err is not None:
                errors += 1
    elapsed = max(timer() - start, 1e-9)
    latencies.sort()
    p = lambda q: latencies[min(len(latencies) - 1, int(len(latencies) * q))] if latencies else 0.0
    return {
        "n": len(jobs),
        "throughput_qps": round(len(jobs) / elapsed, 2),
        "latency_p50_ms": round(p(0.50), 1),
        "latency_p95_ms": round(p(0.95), 1),
        "errors": errors,
    }


def main() -> None:
    import httpx

    def send_fn(text: str) -> None:
        httpx.post("http://localhost:8000/chat", json={"question": text}, timeout=120).raise_for_status()

    res = run_load(send_fn, load_golden(), concurrency=8)
    print(res)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_load_test.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add backend/eval/load_test.py backend/tests/test_load_test.py
git commit -m "feat(eval): optional concurrency/load harness"
```

---

## End-to-end usage (after implementation)

```bash
make up                 # Postgres + Neo4j
make scale-seed         # core @ 5k customers / 50k orders + ~1000 empty distractor tables
make scale-ingest       # build the graph at scale (routing on, fake embeddings)
make eval               # full scorecard.json (routing + answers + latency)
make eval-baseline      # small-catalog baseline for comparison
make scale-teardown     # drop scale_* schemas
```

Compare `scorecard.json` (scaled) against `scorecard-baseline.json` (baseline): a
drop in `routing_hit_rate` or `answer_accuracy` at scale is the signal that the
agents are degrading as the catalog grows. Per project memory, doc-RAG questions
require embeddings present — `make scale-ingest` populates them (fake vectors), so
run eval after ingest, not after a bare `pytest`.

## Notes for the implementer

- **Routing does not use embeddings** (`agent/routing.py:33-48` aggregates
  `search_catalog` keyword hits), so `FAKE_EMBEDDINGS=true` does not distort routing
  metrics — it only lowers doc-RAG semantic quality, which the doc-RAG golden
  questions assert loosely (keyword `contains`).
- **Real OpenAI cost** during `make eval` (not `--routing-only`) comes from the
  per-question `ask()` agent run and the one ranking LLM call per question. Use
  `--routing-only` while iterating; run the full eval for fidelity numbers.
- **Determinism:** the same `SCALE_N_TABLES`/`SCALE_N_APIS`/`RANDOM_SEED` reproduce
  the identical catalog in both `scale-seed` and `scale-ingest` (both call
  `generate_scale_catalog` with `settings.random_seed`), so the schemas created and
  the schemas introspected always agree.
