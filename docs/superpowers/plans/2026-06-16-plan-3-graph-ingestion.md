# Plan 3: Graph Ingestion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the unified semantic + context graph in Neo4j by ingesting all three source types: the structured databases (Postgres + SQLite) and the mock APIs become a NeoCarta **metadata layer** (Database→Schema→Table→Column + `REFERENCES`), the NVIDIA PDFs become a **document/entity layer** (Document→Chunk→Entity with provenance) parsed by liteparse v2, an LLM **glossary layer** (BusinessTerm/Glossary/Category tagged onto columns/endpoints and entities) bridges them, and OpenAI embeddings + vector/full-text indexes make the graph hybrid-searchable.

**Architecture:** A thin `graph/` package wraps the Neo4j driver. The metadata layer is built with the **NeoCarta library** (`neocarta.data_model.rdbms` models + `neocarta.ingest.rdbms.Neo4jRDBMSLoader`), fed by our own extractors that introspect the live SQL databases and the mock-API OpenAPI specs into NeoCarta model objects. The document/entity layer is built directly on the Neo4j driver (NeoCarta's rdbms model has no Document concept). An LLM (`gpt-5.4-mini` via `init_chat_model`) generates BusinessTerms and extracts POLE+O entities. Embeddings use NeoCarta's `OpenAIEmbeddingsConnector` for metadata nodes and a direct OpenAI call + `create_vector_index` for chunks. A single `ingest` orchestrator runs the whole pipeline idempotently.

**Tech Stack:** Python 3.11, `neocarta>=0.7`, `liteparse>=2`, `neo4j` (driver), `openai`, `langchain` (`init_chat_model`) + `langchain-openai`, `psycopg`, stdlib `sqlite3`, `pytest`. Builds on Plan 1 (DBs) and Plan 2 (mock APIs / OpenAPI specs).

**Prerequisites:**
- Plan 1 + Plan 2 merged. Postgres seeded (`make seed`), SQLite DBs built.
- Neo4j running (`make up`) — bolt `bolt://localhost:7687`, auth `neo4j/neocarta123`.
- `OPENAI_API_KEY` set in `backend/.env` for the LLM + embedding tasks. Tasks that call OpenAI **skip** (pytest marker `openai`) when the key is absent, mirroring Plan 1's `postgres` marker; tasks that need Neo4j use a `neo4j` marker + skip-if-unreachable fixture.

This is sub-plan 3 of 5 (Data Foundation → Mock APIs → **Graph Ingestion** → Agent → Web App).

---

## Verified external APIs (do not re-derive — these were confirmed against the installed libraries)

**liteparse 2.0.0**
```python
from liteparse import LiteParse
result = LiteParse().parse("docs/NVIDIAAn_2025.pdf")   # ParseResult
result.text        # full document text (str)
result.num_pages   # int
result.get_page(i) # ParsedPage
```

**NeoCarta 0.7.0** — models in `neocarta.data_model.rdbms`:
- `Database(id, name, platform=None, service=None, description=None, embedding=None)`
- `Schema(id, name, description=None, embedding=None)`
- `Table(id, name, description=None, embedding=None)`
- `Column(id, name, description=None, embedding=None, type=None, nullable=bool, is_primary_key=bool, is_foreign_key=bool)`
- `References(source_column_id, target_column_id, criteria=None)`
- `HasSchema(database_id, schema_id)`, `HasTable(schema_id, table_id)`, `HasColumn(table_id, column_id)`
- `BusinessTerm(id, name, description=None, embedding=None, resource_path=None)`, `Glossary(id,name,...)`, `Category(id,name,...)`, `TaggedWith`, `HasBusinessTerm`, `HasCategory`

Loader `neocarta.ingest.rdbms.Neo4jRDBMSLoader(neo4j_driver, database_name="neo4j")`:
```python
loader.load_database_nodes([Database(...)])
loader.load_schema_nodes([Schema(...)])
loader.load_table_nodes([Table(...)])          # auto-creates name + full-text index
loader.load_column_nodes([Column(...)])
loader.load_has_schema_relationships([HasSchema(...)])
loader.load_has_table_relationships([HasTable(...)])
loader.load_has_column_relationships([HasColumn(...)])
loader.load_references_relationships([References(...)])
loader.load_business_term_nodes([BusinessTerm(...)])
loader.load_column_tagged_with_relationships([TaggedWith(...)])  # see Task 8 for exact fields
```
Embeddings `neocarta.enrichment.embeddings.OpenAIEmbeddingsConnector(neo4j_driver, client=openai.OpenAI(), embedding_model="text-embedding-3-small", dimensions=1536, database_name="neo4j").run()`.
Indexes `neocarta.ingest.indexes.create_vector_index(...)`, `create_full_text_index(...)`.

> Two tasks (Task 2 NeoCarta smoke test, Task 8 glossary) include a short `inspect.signature(...)` step to confirm the exact field names of the `expanded` models (`TaggedWith`, `HasBusinessTerm`) and the `create_vector_index` signature **before** using them, since those were not fully captured during planning. Use what `inspect` reports; do not guess.

---

## File Structure

```
backend/
  pyproject.toml                                  # (modify) add neocarta, liteparse, neo4j, openai, langchain, langchain-openai
  semantic_layer/
    config.py                                     # (modify) add embedding_model, llm_model, neo4j_database
    graph/
      __init__.py
      client.py                                   # neo4j driver factory + health check + reset helper
      schema_ids.py                               # deterministic id helpers (source/table/column ids)
    ingest/
      __init__.py
      sql_extractor.py                            # Postgres + SQLite schema -> NeoCarta models
      api_extractor.py                            # mock-API OpenAPI specs -> NeoCarta models (virtual tables)
      metadata_loader.py                          # load NeoCarta models via Neo4jRDBMSLoader
      doc_parser.py                               # liteparse v2 -> Document/Chunk dicts
      doc_loader.py                               # write Document/Chunk nodes + chunk vector index (neo4j driver)
      entities.py                                 # LLM POLE+O entity extraction -> Entity nodes + MENTIONS
      glossary.py                                 # LLM BusinessTerm generation + TaggedWith bridge
      embeddings.py                               # NeoCarta OpenAIEmbeddingsConnector for metadata + chunk embeddings
      llm.py                                       # init_chat_model wrapper + OpenAI client factory
      pipeline.py                                 # orchestrator: run the full ingest idempotently
  tests/
    test_graph_client.py
    test_neocarta_smoke.py                        # neo4j marker
    test_sql_extractor.py                         # postgres marker (reads live DBs)
    test_api_extractor.py
    test_metadata_loader.py                       # neo4j marker
    test_doc_parser.py
    test_doc_loader.py                            # neo4j marker
    test_entities.py                              # openai marker
    test_glossary.py                              # openai marker
    test_embeddings.py                            # openai + neo4j marker
    test_pipeline.py                              # neo4j + postgres + openai marker (end-to-end)
Makefile                                          # (modify) add `ingest` target
backend/README.md                                 # (modify) document graph ingestion
```

---

## Task 1: Dependencies, config, and the Neo4j graph client

**Files:**
- Modify: `backend/pyproject.toml`
- Modify: `backend/semantic_layer/config.py`
- Create: `backend/semantic_layer/graph/__init__.py` (empty)
- Create: `backend/semantic_layer/graph/client.py`
- Create: `backend/semantic_layer/ingest/__init__.py` (empty)
- Test: `backend/tests/conftest.py` (modify — add `neo4j_driver` fixture)
- Test: `backend/tests/test_graph_client.py`

- [ ] **Step 1: Add dependencies to `backend/pyproject.toml`.** The `dependencies` list must add these entries (keep all existing ones):

```toml
    "neocarta>=0.7",
    "liteparse>=2.0",
    "neo4j>=5.0",
    "openai>=1.40",
    "langchain>=0.3",
    "langchain-openai>=0.2",
```

- [ ] **Step 2: Install:** `cd backend && ./.venv/bin/python -m pip install -e ".[dev]"`. Confirm success. If offline, report BLOCKED.

- [ ] **Step 3: Add fields to `backend/semantic_layer/config.py`.** Add these fields to the `Settings` class (after `random_seed`):

```python
    neo4j_database: str = "neo4j"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    llm_model: str = "openai:gpt-5.4-mini"
    docs_dir: str = "../docs"
```

- [ ] **Step 4: Write the failing test** `backend/tests/test_graph_client.py`

```python
import pytest

from semantic_layer.graph.client import get_driver, ping


@pytest.mark.neo4j
def test_driver_connects_and_pings(neo4j_driver):
    assert ping(neo4j_driver) is True


@pytest.mark.neo4j
def test_reset_graph_clears_nodes(neo4j_driver):
    from semantic_layer.graph.client import reset_graph
    with neo4j_driver.session() as s:
        s.run("CREATE (:Probe {k: 1})")
    reset_graph(neo4j_driver)
    with neo4j_driver.session() as s:
        count = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    assert count == 0
```

- [ ] **Step 5: Add the `neo4j_driver` fixture to `backend/tests/conftest.py`** (append; keep the existing `postgres_dsn` fixture). Also register the new markers in `pyproject.toml` `[tool.pytest.ini_options].markers` list: add `"neo4j: tests requiring the docker neo4j service"` and `"openai: tests requiring OPENAI_API_KEY"`.

```python
@pytest.fixture(scope="session")
def neo4j_driver():
    """Skip neo4j-marked tests if the docker neo4j is not reachable."""
    from semantic_layer.graph.client import get_driver
    try:
        driver = get_driver()
        driver.verify_connectivity()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Neo4j not available: {exc}")
    yield driver
    driver.close()


@pytest.fixture(scope="session")
def require_openai():
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd backend && ./.venv/bin/python -m pytest tests/test_graph_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'semantic_layer.graph.client'`

- [ ] **Step 7: Implement `backend/semantic_layer/graph/client.py`**

```python
"""Neo4j driver factory and small graph utilities."""

from neo4j import Driver, GraphDatabase

from semantic_layer.config import settings


def get_driver() -> Driver:
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


def ping(driver: Driver) -> bool:
    driver.verify_connectivity()
    with driver.session(database=settings.neo4j_database) as session:
        return session.run("RETURN 1 AS ok").single()["ok"] == 1


def reset_graph(driver: Driver) -> None:
    """Delete all nodes and relationships. Used before a full re-ingest."""
    with driver.session(database=settings.neo4j_database) as session:
        session.run("MATCH (n) DETACH DELETE n")
```

- [ ] **Step 8: Run test to verify it passes** (Neo4j must be up: `make up`)

Run: `cd backend && ./.venv/bin/python -m pytest tests/test_graph_client.py -v`
Expected: PASS (2 passed). If Neo4j is down they skip — bring it up and get 2 passed.

- [ ] **Step 9: Commit**

```bash
git add backend/pyproject.toml backend/semantic_layer/config.py backend/semantic_layer/graph backend/semantic_layer/ingest/__init__.py backend/tests/conftest.py backend/tests/test_graph_client.py
git commit -m "feat(graph): neo4j client, ingest deps, and neo4j/openai test fixtures"
```

---

## Task 2: NeoCarta smoke test — verify the loader writes the standardized graph

**Files:**
- Create: `backend/semantic_layer/graph/schema_ids.py`
- Test: `backend/tests/test_neocarta_smoke.py`

This task de-risks the NeoCarta integration end-to-end with a tiny hand-built graph before the real extractors, and confirms the exact `expanded` model fields.

- [ ] **Step 1: Implement `backend/semantic_layer/graph/schema_ids.py`** (deterministic id helpers used everywhere)

```python
"""Deterministic, collision-free id helpers for graph nodes.

Ids are stable strings so re-ingestion MERGEs onto the same nodes.
"""


def database_id(source: str) -> str:
    return f"db:{source}"


def schema_id(source: str, schema: str) -> str:
    return f"schema:{source}.{schema}"


def table_id(source: str, schema: str, table: str) -> str:
    return f"table:{source}.{schema}.{table}"


def column_id(source: str, schema: str, table: str, column: str) -> str:
    return f"col:{source}.{schema}.{table}.{column}"
```

- [ ] **Step 2: Confirm the `expanded` model fields, then write the smoke test** `backend/tests/test_neocarta_smoke.py`

First run this once to see the exact fields (use what it prints in the test if they differ):
`cd backend && ./.venv/bin/python -c "from neocarta.data_model import rdbms as R; import inspect; print('TaggedWith', list(R.TaggedWith.model_fields)); print('HasBusinessTerm', list(R.HasBusinessTerm.model_fields))"`

Then the test:
```python
import pytest

from neocarta.data_model.rdbms import (
    Database, Schema, Table, Column, HasSchema, HasTable, HasColumn, References,
)
from neocarta.ingest.rdbms import Neo4jRDBMSLoader

from semantic_layer.config import settings
from semantic_layer.graph.client import reset_graph


@pytest.mark.neo4j
def test_loader_writes_schema_layer(neo4j_driver):
    reset_graph(neo4j_driver)
    loader = Neo4jRDBMSLoader(neo4j_driver, database_name=settings.neo4j_database)

    loader.load_database_nodes([Database(id="db:test", name="test")])
    loader.load_schema_nodes([Schema(id="schema:test.public", name="public")])
    loader.load_table_nodes([Table(id="table:test.public.t", name="t")])
    loader.load_column_nodes([
        Column(id="col:test.public.t.a", name="a", type="INTEGER",
               nullable=False, is_primary_key=True, is_foreign_key=False),
        Column(id="col:test.public.t.b", name="b", type="INTEGER",
               nullable=True, is_primary_key=False, is_foreign_key=True),
    ])
    loader.load_has_schema_relationships([HasSchema(database_id="db:test", schema_id="schema:test.public")])
    loader.load_has_table_relationships([HasTable(schema_id="schema:test.public", table_id="table:test.public.t")])
    loader.load_has_column_relationships([
        HasColumn(table_id="table:test.public.t", column_id="col:test.public.t.a"),
        HasColumn(table_id="table:test.public.t", column_id="col:test.public.t.b"),
    ])
    loader.load_references_relationships([
        References(source_column_id="col:test.public.t.b", target_column_id="col:test.public.t.a"),
    ])

    with neo4j_driver.session(database=settings.neo4j_database) as s:
        cols = s.run(
            "MATCH (t:Table {id:'table:test.public.t'})-[:HAS_COLUMN]->(c:Column) RETURN count(c) AS c"
        ).single()["c"]
        refs = s.run("MATCH (:Column)-[r:REFERENCES]->(:Column) RETURN count(r) AS c").single()["c"]
    assert cols == 2
    assert refs == 1
```

- [ ] **Step 3: Run** `cd backend && ./.venv/bin/python -m pytest tests/test_neocarta_smoke.py -v`
Expected: PASS (1 passed). If the relationship/label names differ from what the loader produces, inspect the graph in the Neo4j browser and adjust the assertion Cypher to match NeoCarta's actual labels (`Table`, `Column`, `HAS_COLUMN`, `REFERENCES` were confirmed from `RelationshipType`). Do NOT change the loader calls.

- [ ] **Step 4: Commit**

```bash
git add backend/semantic_layer/graph/schema_ids.py backend/tests/test_neocarta_smoke.py
git commit -m "feat(graph): id helpers + NeoCarta loader smoke test"
```

---

## Task 3: SQL schema extractor (Postgres + SQLite → NeoCarta models)

**Files:**
- Create: `backend/semantic_layer/ingest/sql_extractor.py`
- Test: `backend/tests/test_sql_extractor.py`

Produces NeoCarta model objects from the live databases. Pure extraction (returns models; no Neo4j writes).

- [ ] **Step 1: Write the failing test** `backend/tests/test_sql_extractor.py`

```python
import pytest

from semantic_layer.ingest.sql_extractor import extract_postgres, extract_sqlite


@pytest.mark.postgres
def test_extract_postgres_sales_schema(postgres_dsn):
    bundle = extract_postgres(postgres_dsn, source="sales_pg")
    table_names = {t.name for t in bundle.tables}
    assert {"order_line", "product", "region"} <= table_names
    assert len(bundle.tables) == 11
    # foreign keys produce References + is_foreign_key columns
    assert len(bundle.references) >= 10
    assert any(c.is_foreign_key for c in bundle.columns)
    assert any(c.is_primary_key for c in bundle.columns)


def test_extract_sqlite_financials(tmp_path):
    from data.seed_sqlite import seed_all
    seed_all(out_dir=str(tmp_path))
    bundle = extract_sqlite(str(tmp_path / "financials.db"), source="financials")
    names = {t.name for t in bundle.tables}
    assert {"income_statement", "stock_price"} <= names
    assert all(c.type for c in bundle.columns)  # types captured
```

- [ ] **Step 2: Run to verify it fails** — `ModuleNotFoundError: ... sql_extractor`.

- [ ] **Step 3: Implement `backend/semantic_layer/ingest/sql_extractor.py`**

```python
"""Introspect live SQL databases into NeoCarta metadata-layer model objects."""

import sqlite3
from dataclasses import dataclass, field

import psycopg

from neocarta.data_model.rdbms import (
    Database, Schema, Table, Column, HasSchema, HasTable, HasColumn, References,
)

from semantic_layer.graph.schema_ids import (
    database_id, schema_id, table_id, column_id,
)


@dataclass
class SchemaBundle:
    databases: list = field(default_factory=list)
    schemas: list = field(default_factory=list)
    tables: list = field(default_factory=list)
    columns: list = field(default_factory=list)
    has_schema: list = field(default_factory=list)
    has_table: list = field(default_factory=list)
    has_column: list = field(default_factory=list)
    references: list = field(default_factory=list)


def _bundle_for(source: str, schema_name: str, platform: str) -> SchemaBundle:
    b = SchemaBundle()
    b.databases.append(Database(id=database_id(source), name=source, platform=platform))
    b.schemas.append(Schema(id=schema_id(source, schema_name), name=schema_name))
    b.has_schema.append(HasSchema(database_id=database_id(source), schema_id=schema_id(source, schema_name)))
    return b


def extract_postgres(dsn: str, source: str = "sales_pg", schema_name: str = "sales") -> SchemaBundle:
    b = _bundle_for(source, schema_name, platform="postgresql")
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = %s ORDER BY table_name",
            (schema_name,),
        )
        tables = [r[0] for r in cur.fetchall()]

        cur.execute(
            "SELECT table_name, column_name, data_type, is_nullable "
            "FROM information_schema.columns WHERE table_schema = %s",
            (schema_name,),
        )
        col_rows = cur.fetchall()

        # primary keys
        cur.execute(
            """
            SELECT tc.table_name, kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
            WHERE tc.table_schema = %s AND tc.constraint_type = 'PRIMARY KEY'
            """,
            (schema_name,),
        )
        pks = {(t, c) for t, c in cur.fetchall()}

        # foreign keys
        cur.execute(
            """
            SELECT kcu.table_name, kcu.column_name,
                   ccu.table_name AS ref_table, ccu.column_name AS ref_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
              ON tc.constraint_name = ccu.constraint_name AND tc.table_schema = ccu.table_schema
            WHERE tc.table_schema = %s AND tc.constraint_type = 'FOREIGN KEY'
            """,
            (schema_name,),
        )
        fks = cur.fetchall()
    fk_cols = {(t, c) for t, c, _, _ in fks}

    for t in tables:
        b.tables.append(Table(id=table_id(source, schema_name, t), name=t))
        b.has_table.append(HasTable(schema_id=schema_id(source, schema_name), table_id=table_id(source, schema_name, t)))
    for t, c, dtype, nullable in col_rows:
        cid = column_id(source, schema_name, t, c)
        b.columns.append(Column(
            id=cid, name=c, type=dtype, nullable=(nullable == "YES"),
            is_primary_key=(t, c) in pks, is_foreign_key=(t, c) in fk_cols,
        ))
        b.has_column.append(HasColumn(table_id=table_id(source, schema_name, t), column_id=cid))
    for t, c, rt, rc in fks:
        b.references.append(References(
            source_column_id=column_id(source, schema_name, t, c),
            target_column_id=column_id(source, schema_name, rt, rc),
            criteria=f"{t}.{c} -> {rt}.{rc}",
        ))
    return b


def extract_sqlite(db_path: str, source: str, schema_name: str = "main") -> SchemaBundle:
    b = _bundle_for(source, schema_name, platform="sqlite")
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()]
    for t in tables:
        b.tables.append(Table(id=table_id(source, schema_name, t), name=t))
        b.has_table.append(HasTable(schema_id=schema_id(source, schema_name), table_id=table_id(source, schema_name, t)))
        info = con.execute(f"PRAGMA table_info({t})").fetchall()
        fk_list = con.execute(f"PRAGMA foreign_key_list({t})").fetchall()
        fk_cols = {row["from"] for row in fk_list}
        for row in info:
            cid = column_id(source, schema_name, t, row["name"])
            b.columns.append(Column(
                id=cid, name=row["name"], type=row["type"] or "TEXT",
                nullable=(row["notnull"] == 0),
                is_primary_key=bool(row["pk"]), is_foreign_key=row["name"] in fk_cols,
            ))
            b.has_column.append(HasColumn(table_id=table_id(source, schema_name, t), column_id=cid))
        for row in fk_list:
            b.references.append(References(
                source_column_id=column_id(source, schema_name, t, row["from"]),
                target_column_id=column_id(source, schema_name, row["table"], row["to"]),
                criteria=f"{t}.{row['from']} -> {row['table']}.{row['to']}",
            ))
    con.close()
    return b
```

- [ ] **Step 4: Run to verify it passes** (Postgres must be seeded: `make seed`). Expected PASS (2 passed; the postgres test runs, the sqlite test always runs).

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/ingest/sql_extractor.py backend/tests/test_sql_extractor.py
git commit -m "feat(ingest): SQL schema extractor (postgres + sqlite -> NeoCarta models)"
```

---

## Task 4: Metadata loader (write the bundles into Neo4j)

**Files:**
- Create: `backend/semantic_layer/ingest/metadata_loader.py`
- Test: `backend/tests/test_metadata_loader.py`

- [ ] **Step 1: Write the failing test** `backend/tests/test_metadata_loader.py`

```python
import pytest

from semantic_layer.config import settings
from semantic_layer.graph.client import reset_graph
from semantic_layer.ingest.sql_extractor import extract_sqlite
from semantic_layer.ingest.metadata_loader import load_bundle


@pytest.mark.neo4j
def test_load_sqlite_bundle_creates_nodes(neo4j_driver, tmp_path):
    from data.seed_sqlite import seed_all
    seed_all(out_dir=str(tmp_path))
    reset_graph(neo4j_driver)
    bundle = extract_sqlite(str(tmp_path / "org.db"), source="org")
    load_bundle(neo4j_driver, bundle)
    with neo4j_driver.session(database=settings.neo4j_database) as s:
        tables = s.run(
            "MATCH (:Database {id:'db:org'})-[:HAS_SCHEMA]->(:Schema)-[:HAS_TABLE]->(t:Table) "
            "RETURN count(t) AS c"
        ).single()["c"]
        fk = s.run("MATCH (:Column)-[r:REFERENCES]->(:Column) RETURN count(r) AS c").single()["c"]
    assert tables == 3          # department, location, headcount
    assert fk >= 2              # headcount has 2 FKs
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `backend/semantic_layer/ingest/metadata_loader.py`**

```python
"""Load a SchemaBundle into Neo4j via NeoCarta's Neo4jRDBMSLoader (idempotent)."""

from neo4j import Driver

from neocarta.ingest.rdbms import Neo4jRDBMSLoader

from semantic_layer.config import settings
from semantic_layer.ingest.sql_extractor import SchemaBundle


def load_bundle(driver: Driver, bundle: SchemaBundle) -> None:
    loader = Neo4jRDBMSLoader(driver, database_name=settings.neo4j_database)
    if bundle.databases:
        loader.load_database_nodes(bundle.databases, overwrite_existing=True)
    if bundle.schemas:
        loader.load_schema_nodes(bundle.schemas, overwrite_existing=True)
    if bundle.tables:
        loader.load_table_nodes(bundle.tables, overwrite_existing=True)
    if bundle.columns:
        loader.load_column_nodes(bundle.columns, overwrite_existing=True)
    if bundle.has_schema:
        loader.load_has_schema_relationships(bundle.has_schema)
    if bundle.has_table:
        loader.load_has_table_relationships(bundle.has_table)
    if bundle.has_column:
        loader.load_has_column_relationships(bundle.has_column)
    if bundle.references:
        loader.load_references_relationships(bundle.references)
```

- [ ] **Step 4: Run to verify it passes** (Neo4j up). Expected PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/ingest/metadata_loader.py backend/tests/test_metadata_loader.py
git commit -m "feat(ingest): metadata loader writes SchemaBundle via NeoCarta loader"
```

---

## Task 5: API (OpenAPI) extractor — APIs as virtual tables

**Files:**
- Create: `backend/semantic_layer/ingest/api_extractor.py`
- Test: `backend/tests/test_api_extractor.py`

Models each mock API as a NeoCarta `Database`; each GET endpoint as a `Table`; each response-schema field as a `Column`. Reuses `SchemaBundle` and `load_bundle`.

- [ ] **Step 1: Write the failing test** `backend/tests/test_api_extractor.py`

```python
from semantic_layer.apis.app import app
from semantic_layer.ingest.api_extractor import extract_openapi, extract_all_apis


def _spec(prefix: str) -> dict:
    from fastapi.testclient import TestClient
    return TestClient(app).get(f"{prefix}/openapi.json").json()


def test_extract_crm_openapi_to_virtual_tables():
    bundle = extract_openapi(_spec("/crm"), source="crm")
    table_names = {t.name for t in bundle.tables}
    assert "GET /accounts" in table_names
    # Account schema fields become columns on the accounts endpoint table
    col_names = {c.name for c in bundle.columns}
    assert {"account_id", "name", "industry", "region"} <= col_names


def test_extract_all_apis_covers_four_sources():
    bundles = extract_all_apis(_spec, ("crm", "itsm", "partner", "dgx"))
    sources = {db.name for b in bundles for db in b.databases}
    assert sources == {"crm", "itsm", "partner", "dgx"}
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `backend/semantic_layer/ingest/api_extractor.py`**

```python
"""Introspect mock-API OpenAPI specs into NeoCarta metadata models.

API  -> Database (platform='rest-api')
Endpoint (method+path) -> Table
Response-schema property -> Column
"""

from neocarta.data_model.rdbms import (
    Database, Schema, Table, Column, HasSchema, HasTable, HasColumn,
)

from semantic_layer.graph.schema_ids import (
    database_id, schema_id, table_id, column_id,
)
from semantic_layer.ingest.sql_extractor import SchemaBundle

_SCHEMA = "api"


def _resolve_item_schema(operation: dict, components: dict) -> dict | None:
    """Return the object schema of a 200 response (unwrapping array + $ref)."""
    try:
        content = operation["responses"]["200"]["content"]["application/json"]["schema"]
    except KeyError:
        return None
    if content.get("type") == "array":
        content = content.get("items", {})
    ref = content.get("$ref")
    if ref:
        name = ref.split("/")[-1]
        return components.get("schemas", {}).get(name)
    return content if content.get("properties") else None


def extract_openapi(spec: dict, source: str) -> SchemaBundle:
    b = SchemaBundle()
    b.databases.append(Database(id=database_id(source), name=source, platform="rest-api"))
    b.schemas.append(Schema(id=schema_id(source, _SCHEMA), name=_SCHEMA))
    b.has_schema.append(HasSchema(database_id=database_id(source), schema_id=schema_id(source, _SCHEMA)))

    components = spec.get("components", {})
    for path, methods in spec.get("paths", {}).items():
        for method, operation in methods.items():
            endpoint = f"{method.upper()} {path}"
            tid = table_id(source, _SCHEMA, endpoint)
            b.tables.append(Table(id=tid, name=endpoint, description=operation.get("summary")))
            b.has_table.append(HasTable(schema_id=schema_id(source, _SCHEMA), table_id=tid))
            item = _resolve_item_schema(operation, components)
            if not item:
                continue
            for prop, meta in item.get("properties", {}).items():
                cid = column_id(source, _SCHEMA, endpoint, prop)
                b.columns.append(Column(
                    id=cid, name=prop, type=meta.get("type", "string"),
                    nullable=True, is_primary_key=False,
                    is_foreign_key=prop.endswith("_id"),
                ))
                b.has_column.append(HasColumn(table_id=tid, column_id=cid))
    return b


def extract_all_apis(spec_getter, sources) -> list:
    """spec_getter(prefix) -> openapi dict. sources: iterable of api names."""
    return [extract_openapi(spec_getter(f"/{s}"), source=s) for s in sources]
```

- [ ] **Step 4: Run to verify it passes.** Expected PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/ingest/api_extractor.py backend/tests/test_api_extractor.py
git commit -m "feat(ingest): OpenAPI extractor models mock APIs as virtual tables"
```

---

## Task 6: Document parsing + loading (liteparse v2 → Document/Chunk nodes)

**Files:**
- Create: `backend/semantic_layer/ingest/doc_parser.py`
- Create: `backend/semantic_layer/ingest/doc_loader.py`
- Test: `backend/tests/test_doc_parser.py`
- Test: `backend/tests/test_doc_loader.py`

- [ ] **Step 1: Write the failing test** `backend/tests/test_doc_parser.py`

```python
from pathlib import Path

from semantic_layer.ingest.doc_parser import parse_document, chunk_text

DOCS = Path(__file__).resolve().parents[2] / "docs"


def test_chunk_text_splits_with_overlap():
    chunks = chunk_text("abcdefghij", size=4, overlap=1)
    assert chunks[0] == "abcd"
    assert all(len(c) <= 4 for c in chunks)
    assert "".join(c[: 4 - 1] if i else c for i, c in enumerate(chunks)).startswith("abc")


def test_parse_real_pdf_returns_chunks():
    pdf = DOCS / "NVIDIAAn_2025.pdf"
    doc = parse_document(str(pdf))
    assert doc["doc_id"]
    assert doc["num_pages"] > 0
    assert len(doc["chunks"]) > 0
    assert any("NVIDIA" in c["text"] for c in doc["chunks"])
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `backend/semantic_layer/ingest/doc_parser.py`**

```python
"""Parse PDFs with liteparse v2 and split into overlapping chunks."""

from pathlib import Path

from liteparse import LiteParse


def chunk_text(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        chunks.append(text[start : start + size])
        start += step
    return chunks


def parse_document(path: str, size: int = 1200, overlap: int = 150) -> dict:
    result = LiteParse().parse(path)
    doc_id = f"doc:{Path(path).stem}"
    pieces = chunk_text(result.text, size=size, overlap=overlap)
    chunks = [
        {"chunk_id": f"{doc_id}:chunk:{i}", "doc_id": doc_id, "ordinal": i, "text": piece}
        for i, piece in enumerate(pieces)
    ]
    return {
        "doc_id": doc_id,
        "title": Path(path).stem,
        "path": str(path),
        "num_pages": result.num_pages,
        "chunks": chunks,
    }
```

- [ ] **Step 4: Run to verify it passes** (parses the real PDF; takes a few seconds incl. OCR). Expected PASS (2 passed).

- [ ] **Step 5: Write the failing test** `backend/tests/test_doc_loader.py`

```python
import pytest

from semantic_layer.config import settings
from semantic_layer.graph.client import reset_graph
from semantic_layer.ingest.doc_loader import load_document


@pytest.mark.neo4j
def test_load_document_creates_doc_and_chunks(neo4j_driver):
    reset_graph(neo4j_driver)
    doc = {
        "doc_id": "doc:sample", "title": "Sample", "path": "/tmp/sample.pdf", "num_pages": 1,
        "chunks": [
            {"chunk_id": "doc:sample:chunk:0", "doc_id": "doc:sample", "ordinal": 0, "text": "Blackwell GPU"},
            {"chunk_id": "doc:sample:chunk:1", "doc_id": "doc:sample", "ordinal": 1, "text": "Data Center revenue"},
        ],
    }
    load_document(neo4j_driver, doc)
    with neo4j_driver.session(database=settings.neo4j_database) as s:
        n = s.run("MATCH (:Document {id:'doc:sample'})-[:HAS_CHUNK]->(c:Chunk) RETURN count(c) AS c").single()["c"]
    assert n == 2
    # idempotent: second load does not duplicate
    load_document(neo4j_driver, doc)
    with neo4j_driver.session(database=settings.neo4j_database) as s:
        n2 = s.run("MATCH (:Document {id:'doc:sample'})-[:HAS_CHUNK]->(c:Chunk) RETURN count(c) AS c").single()["c"]
    assert n2 == 2
```

- [ ] **Step 6: Implement `backend/semantic_layer/ingest/doc_loader.py`**

```python
"""Write Document and Chunk nodes to Neo4j (idempotent via MERGE)."""

from neo4j import Driver

from semantic_layer.config import settings

_DOC_CYPHER = """
MERGE (d:Document {id: $doc_id})
SET d.title = $title, d.path = $path, d.num_pages = $num_pages
WITH d
UNWIND $chunks AS ch
MERGE (c:Chunk {id: ch.chunk_id})
SET c.text = ch.text, c.ordinal = ch.ordinal, c.doc_id = ch.doc_id
MERGE (d)-[:HAS_CHUNK]->(c)
"""


def load_document(driver: Driver, doc: dict) -> None:
    with driver.session(database=settings.neo4j_database) as session:
        session.run(
            _DOC_CYPHER,
            doc_id=doc["doc_id"], title=doc["title"], path=doc["path"],
            num_pages=doc["num_pages"], chunks=doc["chunks"],
        )
```

- [ ] **Step 7: Run both doc tests.** Expected: `test_doc_loader.py` PASS (1 passed) with Neo4j up.

- [ ] **Step 8: Commit**

```bash
git add backend/semantic_layer/ingest/doc_parser.py backend/semantic_layer/ingest/doc_loader.py backend/tests/test_doc_parser.py backend/tests/test_doc_loader.py
git commit -m "feat(ingest): liteparse v2 doc parsing + Document/Chunk graph loader"
```

---

## Task 7: LLM helper + POLE+O entity extraction

**Files:**
- Create: `backend/semantic_layer/ingest/llm.py`
- Create: `backend/semantic_layer/ingest/entities.py`
- Test: `backend/tests/test_entities.py`

- [ ] **Step 1: Implement `backend/semantic_layer/ingest/llm.py`**

```python
"""LLM + OpenAI client factories (model ids from config)."""

import openai
from langchain.chat_models import init_chat_model

from semantic_layer.config import settings


def get_chat_model():
    return init_chat_model(settings.llm_model)


def get_openai_client() -> openai.OpenAI:
    return openai.OpenAI()
```

- [ ] **Step 2: Write the failing test** `backend/tests/test_entities.py`

```python
import pytest

from semantic_layer.ingest.entities import extract_entities, POLE_LABELS


@pytest.mark.openai
def test_extract_entities_finds_nvidia_org(require_openai):
    text = (
        "NVIDIA announced record Data Center revenue driven by the Blackwell "
        "architecture. CEO Jensen Huang highlighted demand in the United States."
    )
    ents = extract_entities(text)
    assert len(ents) > 0
    assert all(e["label"] in POLE_LABELS for e in ents)
    names = {e["name"].lower() for e in ents}
    assert any("nvidia" in n for n in names)
```

- [ ] **Step 3: Implement `backend/semantic_layer/ingest/entities.py`**

```python
"""Extract POLE+O entities (Person, Org, Location, Event, Object) from text via LLM."""

import json

from semantic_layer.ingest.llm import get_chat_model

POLE_LABELS = {"Person", "Org", "Location", "Event", "Object"}

_PROMPT = (
    "Extract named entities from the text. Return ONLY a JSON array of objects "
    'with keys "name" and "label". label must be one of: '
    "Person, Org, Location, Event, Object. Deduplicate by name. Text:\n\n{text}"
)


def extract_entities(text: str) -> list[dict]:
    model = get_chat_model()
    resp = model.invoke(_PROMPT.format(text=text[:6000]))
    content = resp.content if hasattr(resp, "content") else str(resp)
    content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        return []
    out = []
    seen = set()
    for item in raw:
        name = (item.get("name") or "").strip()
        label = (item.get("label") or "").strip().capitalize()
        if name and label in POLE_LABELS and name.lower() not in seen:
            seen.add(name.lower())
            out.append({"name": name, "label": label})
    return out
```

- [ ] **Step 4: Run** `cd backend && ./.venv/bin/python -m pytest tests/test_entities.py -v` (skips without `OPENAI_API_KEY`; with a key, PASS 1).

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/ingest/llm.py backend/semantic_layer/ingest/entities.py backend/tests/test_entities.py
git commit -m "feat(ingest): LLM helper + POLE+O entity extraction"
```

---

## Task 8: LLM glossary + bridge (BusinessTerms tagged onto columns/endpoints)

**Files:**
- Create: `backend/semantic_layer/ingest/glossary.py`
- Test: `backend/tests/test_glossary.py`

This builds the bridge layer: LLM-generated `BusinessTerm` nodes linked to metadata columns via NeoCarta `TaggedWith`. First confirm the `TaggedWith` field names (Task 2 printed them).

- [ ] **Step 1: Confirm `TaggedWith` fields** — run:
`cd backend && ./.venv/bin/python -c "from neocarta.data_model.rdbms import TaggedWith; print(list(TaggedWith.model_fields))"`
Use the reported field names (commonly `business_term_id` + `column_id`/`table_id`) in the implementation below; adjust the `TaggedWith(...)` kwargs if they differ.

- [ ] **Step 2: Write the failing test** `backend/tests/test_glossary.py`

```python
import pytest

from semantic_layer.ingest.glossary import generate_business_terms


@pytest.mark.openai
def test_generate_business_terms_for_columns(require_openai):
    columns = [
        {"column_id": "col:sales_pg.sales.order_line.amount", "name": "amount", "table": "order_line"},
        {"column_id": "col:sales_pg.sales.segment.name", "name": "name", "table": "segment"},
    ]
    terms = generate_business_terms(columns)
    assert len(terms) > 0
    for t in terms:
        assert t["name"] and t["description"]
        assert t["column_id"] in {c["column_id"] for c in columns}
```

- [ ] **Step 3: Implement `backend/semantic_layer/ingest/glossary.py`**

```python
"""Generate business glossary terms from schema columns and tag them onto columns."""

import json

from neo4j import Driver

from neocarta.data_model.rdbms import BusinessTerm
from neocarta.ingest.rdbms import Neo4jRDBMSLoader

from semantic_layer.config import settings
from semantic_layer.ingest.llm import get_chat_model

_PROMPT = (
    "You are a data catalog expert for an NVIDIA enterprise. For each column below, "
    "produce a concise business term and a one-sentence business definition. "
    'Return ONLY a JSON array of {"column_id","name","description"}. Columns:\n\n{cols}'
)


def generate_business_terms(columns: list[dict]) -> list[dict]:
    model = get_chat_model()
    payload = "\n".join(f'- {c["column_id"]} (column "{c["name"]}" on table {c["table"]})' for c in columns)
    resp = model.invoke(_PROMPT.format(cols=payload))
    content = resp.content if hasattr(resp, "content") else str(resp)
    content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        return []
    valid_ids = {c["column_id"] for c in columns}
    return [
        {"column_id": t["column_id"], "name": t["name"].strip(), "description": t["description"].strip()}
        for t in raw
        if t.get("column_id") in valid_ids and t.get("name") and t.get("description")
    ]


def load_business_terms(driver: Driver, terms: list[dict]) -> None:
    """Create BusinessTerm nodes and TAGGED_WITH edges to their columns."""
    loader = Neo4jRDBMSLoader(driver, database_name=settings.neo4j_database)
    bt_nodes = [
        BusinessTerm(id=f"term:{i}", name=t["name"], description=t["description"])
        for i, t in enumerate(terms)
    ]
    loader.load_business_term_nodes(bt_nodes, overwrite_existing=True)
    # TAGGED_WITH: business term -> column. Use the field names confirmed in Step 1.
    with driver.session(database=settings.neo4j_database) as session:
        session.run(
            """
            UNWIND $rows AS row
            MATCH (bt:BusinessTerm {id: row.term_id})
            MATCH (c:Column {id: row.column_id})
            MERGE (c)-[:TAGGED_WITH]->(bt)
            """,
            rows=[{"term_id": f"term:{i}", "column_id": t["column_id"]} for i, t in enumerate(terms)],
        )
```

> Note: `load_business_terms` writes the `TAGGED_WITH` edge with explicit Cypher (rather than the loader's `load_column_tagged_with_relationships`) to avoid depending on the exact `TaggedWith` field names; the loader is still used for the nodes. If Step 1 shows `TaggedWith(business_term_id=, column_id=)`, you may switch to `loader.load_column_tagged_with_relationships([...])` instead — either is acceptable as long as the test passes.

- [ ] **Step 4: Run** (skips without key; with key PASS 1).

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/ingest/glossary.py backend/tests/test_glossary.py
git commit -m "feat(ingest): LLM business glossary + TAGGED_WITH bridge to columns"
```

---

## Task 9: Embeddings + vector indexes

**Files:**
- Create: `backend/semantic_layer/ingest/embeddings.py`
- Test: `backend/tests/test_embeddings.py`

- [ ] **Step 1: Confirm `create_vector_index` signature** — run:
`cd backend && ./.venv/bin/python -c "import inspect, neocarta.ingest.indexes as ix; print(inspect.signature(ix.create_vector_index))"`
Use the reported parameter names in the implementation (the call below assumes `(driver, label, property, index_name, dimensions, similarity_function)` — adjust to the real signature).

- [ ] **Step 2: Write the failing test** `backend/tests/test_embeddings.py`

```python
import pytest

from semantic_layer.config import settings
from semantic_layer.graph.client import reset_graph
from semantic_layer.ingest.embeddings import embed_chunks


@pytest.mark.neo4j
@pytest.mark.openai
def test_embed_chunks_sets_vectors(neo4j_driver, require_openai):
    reset_graph(neo4j_driver)
    with neo4j_driver.session(database=settings.neo4j_database) as s:
        s.run("CREATE (:Chunk {id:'c1', text:'NVIDIA Blackwell Data Center revenue'})")
    embed_chunks(neo4j_driver)
    with neo4j_driver.session(database=settings.neo4j_database) as s:
        dim = s.run(
            "MATCH (c:Chunk {id:'c1'}) RETURN size(c.embedding) AS d"
        ).single()["d"]
    assert dim == settings.embedding_dimensions
```

- [ ] **Step 3: Implement `backend/semantic_layer/ingest/embeddings.py`**

```python
"""Create embeddings + vector indexes for chunks and metadata nodes."""

from neo4j import Driver

import neocarta.ingest.indexes as nc_indexes
from neocarta.enrichment.embeddings import OpenAIEmbeddingsConnector

from semantic_layer.config import settings
from semantic_layer.ingest.llm import get_openai_client


def embed_chunks(driver: Driver, batch: int = 64) -> None:
    """Embed Chunk.text into Chunk.embedding and ensure a vector index exists."""
    client = get_openai_client()
    with driver.session(database=settings.neo4j_database) as session:
        rows = session.run(
            "MATCH (c:Chunk) WHERE c.embedding IS NULL RETURN c.id AS id, c.text AS text"
        ).data()
        for i in range(0, len(rows), batch):
            window = rows[i : i + batch]
            vectors = client.embeddings.create(
                model=settings.embedding_model,
                input=[r["text"] for r in window],
                dimensions=settings.embedding_dimensions,
            ).data
            session.run(
                """
                UNWIND $rows AS row
                MATCH (c:Chunk {id: row.id})
                CALL db.create.setNodeVectorProperty(c, 'embedding', row.vec)
                """,
                rows=[{"id": w["id"], "vec": v.embedding} for w, v in zip(window, vectors)],
            )
    _ensure_chunk_vector_index(driver)


def _ensure_chunk_vector_index(driver: Driver) -> None:
    # Use NeoCarta's helper; adjust kwargs to the signature confirmed in Step 1.
    try:
        nc_indexes.create_vector_index(
            driver,
            label="Chunk",
            property="embedding",
            index_name="chunk_embeddings",
            dimensions=settings.embedding_dimensions,
            similarity_function="cosine",
        )
    except TypeError:
        # Fallback to raw Cypher if NeoCarta's signature differs.
        with driver.session(database=settings.neo4j_database) as session:
            session.run(
                f"""
                CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {{indexConfig: {{
                  `vector.dimensions`: {settings.embedding_dimensions},
                  `vector.similarity_function`: 'cosine'
                }}}}
                """
            )


def embed_metadata_nodes(driver: Driver) -> None:
    """Embed Table/Column/BusinessTerm nodes via NeoCarta's OpenAI connector."""
    connector = OpenAIEmbeddingsConnector(
        driver,
        client=get_openai_client(),
        embedding_model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        database_name=settings.neo4j_database,
    )
    connector.run()
```

- [ ] **Step 4: Run** (skips without Neo4j+key; with both, PASS 1).

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/ingest/embeddings.py backend/tests/test_embeddings.py
git commit -m "feat(ingest): chunk + metadata embeddings with vector indexes"
```

---

## Task 10: Pipeline orchestrator, Makefile target, README, end-to-end test

**Files:**
- Create: `backend/semantic_layer/ingest/pipeline.py`
- Test: `backend/tests/test_pipeline.py`
- Modify: `Makefile`
- Modify: `backend/README.md`

- [ ] **Step 1: Implement `backend/semantic_layer/ingest/pipeline.py`**

```python
"""Run the full graph-ingestion pipeline idempotently.

Order: reset -> SQL metadata -> API metadata -> documents -> entities ->
glossary bridge -> embeddings.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from semantic_layer.config import settings
from semantic_layer.graph.client import get_driver, reset_graph
from semantic_layer.apis.app import app
from semantic_layer.ingest.sql_extractor import extract_postgres, extract_sqlite
from semantic_layer.ingest.api_extractor import extract_all_apis
from semantic_layer.ingest.metadata_loader import load_bundle
from semantic_layer.ingest.doc_parser import parse_document
from semantic_layer.ingest.doc_loader import load_document


def _api_spec_getter():
    client = TestClient(app)
    return lambda prefix: client.get(f"{prefix}/openapi.json").json()


def run_ingest(*, with_llm: bool = True, reset: bool = True) -> dict:
    driver = get_driver()
    counts = {}
    try:
        if reset:
            reset_graph(driver)

        # 1. SQL metadata layer
        sqlite_dir = Path(settings.sqlite_dir)
        bundles = [
            extract_postgres(settings.postgres_dsn, source="sales_pg"),
            extract_sqlite(str(sqlite_dir / "financials.db"), source="financials"),
            extract_sqlite(str(sqlite_dir / "org.db"), source="org"),
        ]
        # 2. API metadata layer
        bundles += extract_all_apis(_api_spec_getter(), ("crm", "itsm", "partner", "dgx"))
        for b in bundles:
            load_bundle(driver, b)
        counts["sources"] = len(bundles)

        # 3. Documents
        docs_dir = Path(settings.docs_dir)
        pdfs = sorted(docs_dir.glob("*.pdf"))
        for pdf in pdfs:
            doc = parse_document(str(pdf))
            load_document(driver, doc)
        counts["documents"] = len(pdfs)

        if with_llm:
            _run_llm_stages(driver, bundles)
        return counts
    finally:
        driver.close()


def _run_llm_stages(driver, bundles) -> None:
    from semantic_layer.ingest.entities import extract_entities
    from semantic_layer.ingest.glossary import generate_business_terms, load_business_terms
    from semantic_layer.ingest.embeddings import embed_chunks, embed_metadata_nodes

    # Glossary bridge over a sample of columns from the structured sources.
    columns = [
        {"column_id": c.id, "name": c.name, "table": c.id.split(".")[-2]}
        for b in bundles for c in b.columns
    ][:60]
    if columns:
        load_business_terms(driver, generate_business_terms(columns))

    # Entities from chunks -> Entity nodes + MENTIONS (provenance: chunk id).
    with driver.session(database=settings.neo4j_database) as session:
        chunk_rows = session.run(
            "MATCH (c:Chunk) RETURN c.id AS id, c.text AS text ORDER BY c.id LIMIT 40"
        ).data()
    for row in chunk_rows:
        for ent in extract_entities(row["text"]):
            with driver.session(database=settings.neo4j_database) as session:
                session.run(
                    """
                    MERGE (e:Entity {name: $name}) SET e.label = $label
                    WITH e
                    MATCH (c:Chunk {id: $chunk_id})
                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    name=ent["name"], label=ent["label"], chunk_id=row["id"],
                )

    embed_chunks(driver)
    embed_metadata_nodes(driver)


if __name__ == "__main__":
    print(run_ingest())
```

- [ ] **Step 2: Write the failing test** `backend/tests/test_pipeline.py`

```python
import pytest

from semantic_layer.config import settings
from semantic_layer.graph.client import get_driver
from semantic_layer.ingest.pipeline import run_ingest


@pytest.mark.neo4j
@pytest.mark.postgres
def test_metadata_and_docs_ingest_without_llm(neo4j_driver):
    # LLM stages are skipped so this runs without an OpenAI key.
    counts = run_ingest(with_llm=False, reset=True)
    assert counts["sources"] == 7          # 3 DBs + 4 APIs
    assert counts["documents"] >= 1
    with get_driver().session(database=settings.neo4j_database) as s:
        tables = s.run("MATCH (t:Table) RETURN count(t) AS c").single()["c"]
        chunks = s.run("MATCH (c:Chunk) RETURN count(c) AS c").single()["c"]
        refs = s.run("MATCH (:Column)-[r:REFERENCES]->(:Column) RETURN count(r) AS c").single()["c"]
    assert tables >= 11           # 11 sales tables + API endpoints + other DB tables
    assert chunks > 0
    assert refs >= 10             # sales FKs
```

- [ ] **Step 3: Run** (Neo4j + Postgres up; no key needed for `with_llm=False`). Expected PASS (1 passed).

- [ ] **Step 4: Add the `ingest` target to the repo-root `Makefile`.** Add `ingest` to `.PHONY`, then add (TAB-indented):

```makefile
ingest:
	cd backend && python -m semantic_layer.ingest.pipeline
```

Verify: `make -n ingest` prints the command with no error.

- [ ] **Step 5: Append a Graph Ingestion section to `backend/README.md`** documenting: prerequisites (`make up`, `make seed`, `OPENAI_API_KEY`), `make ingest`, the three graph layers (metadata via NeoCarta, document/entity, glossary bridge), and that re-running is idempotent. Include the layer diagram in prose.

- [ ] **Step 6: Run the full suite** `cd backend && ./.venv/bin/python -m pytest -q` and paste the summary. neo4j/openai/postgres-marked tests run or skip per environment; all non-marked tests must pass.

- [ ] **Step 7: Commit**

```bash
git add backend/semantic_layer/ingest/pipeline.py backend/tests/test_pipeline.py Makefile backend/README.md
git commit -m "feat(ingest): full graph-ingestion pipeline orchestrator + make ingest"
```

---

## Self-Review

**Spec coverage (Plan 3 scope):** metadata layer via NeoCarta library — SQL extractor (Task 3) + API/OpenAPI extractor (Task 5) + loader (Tasks 2, 4) ✓; `REFERENCES` FK edges (Tasks 3, 4) ✓; document layer via liteparse v2 + chunking (Task 6) ✓; POLE+O entity layer with provenance (Task 7 + pipeline `MENTIONS` from chunk) ✓; LLM glossary BusinessTerms + `TAGGED_WITH` bridge (Task 8) ✓; embeddings + vector/full-text indexes (Task 9; loader auto-creates full-text indexes) ✓; idempotent re-ingest (`MERGE` + `overwrite_existing=True` + `reset`) ✓; orchestrator + `make ingest` (Task 10) ✓. The agent that consumes this graph (semantic tools + deepagents) is Plan 4.

**External-API honesty:** all NeoCarta/liteparse calls use interfaces verified against the installed packages during planning; the three genuinely-uncertain spots (`TaggedWith` fields, `create_vector_index` signature, NeoCarta's exact relationship labels) each have an explicit `inspect`/confirm step before use plus a raw-Cypher fallback — no guessed APIs are committed.

**Marker discipline:** tests needing services are gated — `neo4j` (skip if bolt unreachable), `postgres` (existing), `openai` (skip without key). The end-to-end pipeline test runs with `with_llm=False` so CI without an OpenAI key still verifies metadata + document ingestion.

**Type/name consistency:** `SchemaBundle` (Task 3) is produced by both `extract_postgres`/`extract_sqlite` (Task 3) and `extract_openapi` (Task 5) and consumed by `load_bundle` (Task 4) and `pipeline` (Task 10). `database_id/schema_id/table_id/column_id` (Task 2) are used identically across extractors. `parse_document` output keys (`doc_id,title,path,num_pages,chunks[{chunk_id,doc_id,ordinal,text}]`) (Task 6) match `load_document`'s Cypher params (Task 6) and the pipeline. `settings.embedding_dimensions` (Task 1) is used by both `embed_chunks` and the index creation (Task 9). Config additions in Task 1 (`neo4j_database`, `embedding_model`, `embedding_dimensions`, `llm_model`, `docs_dir`) are referenced consistently throughout.

**Scope check:** one coherent subsystem (build the graph). 10 tasks, each independently testable. Larger than Plans 1–2 but cohesive; the agent/UI remain separate plans.
```
