# Plan 4: deepagents Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the agent that answers natural-language questions uniformly across the whole semantic layer. A **deepagents** orchestrator uses graph-backed semantic tools (catalog search, schema lookup, **graph join-path discovery**, document vector search) to route each question and delegates execution to three subagents — **sql** (grounded text-to-SQL over Postgres/SQLite), **api** (REST calls to the mock enterprise APIs), and **doc** (RAG over the document chunks) — then synthesizes an answer with provenance.

**Architecture:** Tools are LangChain `@tool` functions over the Plan 3 Neo4j graph + the underlying data sources. The orchestrator is built with `create_deep_agent(model, tools, system_prompt, subagents=[...])` (deepagents 0.6.x) on `gpt-5.4-mini` via `init_chat_model`. The graph supplies routing/grounding (which source, which tables, which join path); the subagents fetch real values (SQL rows, API JSON, document passages). The signature capability is `get_join_path`, which turns a deep 6+-table join into a graph traversal over `REFERENCES` edges that the sql subagent then executes.

**Tech Stack:** Python 3.11, `deepagents>=0.6`, `langchain` (`init_chat_model`), `langchain-community` (`SQLDatabase`), `neo4j`, `openai` (query embeddings), `psycopg`, stdlib `sqlite3`, FastAPI `TestClient` (for API calls), `pytest`. Builds on Plan 1 (DBs), Plan 2 (mock APIs), Plan 3 (the graph).

**Prerequisites:** Plans 1–3 merged. Neo4j + Postgres up; databases seeded; **the graph ingested** (`make ingest`, or `run_ingest(with_llm=False)` for metadata+docs). `OPENAI_API_KEY` in `backend/.env` (agent + query embeddings). Tests gate on `neo4j` / `postgres` / `openai` markers and skip when unavailable.

This is sub-plan 4 of 5 (Data Foundation → Mock APIs → Graph Ingestion → **Agent** → Web App).

---

## Verified external APIs (confirmed against installed packages — do not re-derive)

**deepagents 0.6.10**
```python
from deepagents import create_deep_agent
agent = create_deep_agent(
    model="openai:gpt-5.4-mini",          # or a BaseChatModel
    tools=[tool_a, tool_b],               # orchestrator tools
    system_prompt="...",
    subagents=[                            # each is a SubAgent dict
        {"name": "sql", "description": "...", "system_prompt": "...", "tools": [run_sql], "model": "openai:gpt-5.4-mini"},
    ],
)
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
final_text = result["messages"][-1].content
```
`SubAgent` required keys: `name`, `description`, `system_prompt`; optional: `tools`, `model`, … The orchestrator gets a built-in `task` tool to delegate to subagents by name, plus `write_todos` planning and a virtual filesystem.

**LangChain tools:** `from langchain_core.tools import tool` (decorator → BaseTool). Tool return values are strings the model reads; return JSON strings or readable text.

**Graph facts (from Plan 3):** labels `Database/Schema/Table/Column/BusinessTerm/Document/Chunk/Entity`; rels `HAS_SCHEMA/HAS_TABLE/HAS_COLUMN/REFERENCES/TAGGED_WITH/HAS_CHUNK/MENTIONS`. `Table.id = "table:{source}.{schema}.{table}"`, `Column.id = "col:{source}.{schema}.{table}.{column}"`, `Column` has `name,type,is_primary_key,is_foreign_key`. `Chunk.embedding` is indexed by the vector index named **`chunk_embeddings`** (1536-d, cosine). Sources: `sales_pg` (postgres, schema `sales`), `financials` + `org` (sqlite, schema `main`), and `crm/itsm/partner/dgx` (platform `REST-API`, schema `api`).

---

## File Structure

```
backend/
  pyproject.toml                              # (modify) add deepagents, langchain-community
  semantic_layer/
    config.py                                 # (modify) add agent_max_rows
    agent/
      __init__.py
      driver.py                               # cached neo4j driver for tools
      graph_tools.py                          # list_sources, get_table_schema, get_join_path, search_catalog
      doc_tools.py                            # search_documents (chunk vector search)
      sql_tools.py                            # run_sql (read-only, per-source routing)
      api_tools.py                            # call_api (TestClient against mock APIs)
      build.py                                # subagents + create_deep_agent + ask()
      cli.py                                  # `python -m semantic_layer.agent.cli "question"`
  tests/
    conftest.py                               # (modify) add session `ingested_graph` fixture
    test_agent_graph_tools.py                 # neo4j (+ uses ingested graph)
    test_agent_join_path.py                   # neo4j
    test_agent_catalog.py                     # neo4j
    test_agent_doc_tools.py                   # neo4j + openai
    test_agent_sql_tools.py                   # postgres
    test_agent_api_tools.py
    test_agent_end_to_end.py                  # neo4j + postgres + openai (golden questions)
Makefile                                      # (modify) add `ask` target
backend/README.md                             # (modify) document the agent
```

---

## Task 1: Scaffold — deps, config, driver helper, ingested-graph fixture

**Files:**
- Modify: `backend/pyproject.toml`
- Modify: `backend/semantic_layer/config.py`
- Create: `backend/semantic_layer/agent/__init__.py` (empty)
- Create: `backend/semantic_layer/agent/driver.py`
- Modify: `backend/tests/conftest.py`
- Test: `backend/tests/test_agent_scaffold.py`

- [ ] **Step 1: Add deps to `pyproject.toml`** dependencies list: `"deepagents>=0.6"`, `"langchain-community>=0.3"`. Install: `cd backend && ./.venv/bin/python -m pip install -e ".[dev]"`.

- [ ] **Step 2: Add to `config.py` Settings** (after `docs_dir`): `agent_max_rows: int = 100`.

- [ ] **Step 3: Implement `backend/semantic_layer/agent/driver.py`**

```python
"""Process-wide cached Neo4j driver for agent tools."""

from functools import lru_cache

from neo4j import Driver

from semantic_layer.graph.client import get_driver


@lru_cache
def driver() -> Driver:
    return get_driver()
```

- [ ] **Step 4: Add a session fixture to `backend/tests/conftest.py`** (append):

```python
@pytest.fixture(scope="session")
def ingested_graph(neo4j_driver, postgres_dsn):
    """Build the metadata + document graph once (no LLM) for agent tool tests."""
    from semantic_layer.ingest.pipeline import run_ingest
    run_ingest(with_llm=False, reset=True)
    return neo4j_driver
```

- [ ] **Step 5: Write + pass a scaffold test** `backend/tests/test_agent_scaffold.py`

```python
import pytest

from semantic_layer.agent.driver import driver


@pytest.mark.neo4j
def test_driver_is_cached(neo4j_driver):
    assert driver() is driver()
```

Run: `cd backend && ./.venv/bin/python -m pytest tests/test_agent_scaffold.py -v` → 1 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/pyproject.toml backend/semantic_layer/config.py backend/semantic_layer/agent/__init__.py backend/semantic_layer/agent/driver.py backend/tests/conftest.py backend/tests/test_agent_scaffold.py
git commit -m "feat(agent): scaffold deps, config, driver helper, ingested-graph fixture"
```

---

## Task 2: Catalog tools — `list_sources` and `get_table_schema`

**Files:**
- Create: `backend/semantic_layer/agent/graph_tools.py`
- Test: `backend/tests/test_agent_graph_tools.py`

- [ ] **Step 1: Write the failing test** `backend/tests/test_agent_graph_tools.py`

```python
import json

import pytest

from semantic_layer.agent.graph_tools import list_sources, get_table_schema


@pytest.mark.neo4j
def test_list_sources_includes_db_and_api(ingested_graph):
    data = json.loads(list_sources.invoke({}))
    names = {s["name"] for s in data}
    assert {"sales_pg", "financials", "org", "crm", "itsm", "partner", "dgx"} <= names
    kinds = {s["name"]: s["kind"] for s in data}
    assert kinds["sales_pg"] == "sql"
    assert kinds["crm"] == "api"


@pytest.mark.neo4j
def test_get_table_schema_for_order_line(ingested_graph):
    schema = json.loads(get_table_schema.invoke({"table_id": "table:sales_pg.sales.order_line"}))
    assert schema["sql_reference"] == "sales.order_line"
    assert schema["source"] == "sales_pg"
    col_names = {c["name"] for c in schema["columns"]}
    assert {"order_id", "product_id", "amount"} <= col_names
    assert any(c["is_foreign_key"] for c in schema["columns"])
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `backend/semantic_layer/agent/graph_tools.py`** (this task adds `list_sources` + `get_table_schema`; `get_join_path` and `search_catalog` are added in Tasks 3–4 into this same file)

```python
"""Graph-backed semantic tools: source catalog, schema lookup, join paths, search."""

import json

from langchain_core.tools import tool

from semantic_layer.agent.driver import driver
from semantic_layer.config import settings

_SQL_PLATFORMS = {"POSTGRESQL", "SQLITE"}


def _sql_reference(table_id: str) -> str:
    # table:{source}.{schema}.{table} ; sqlite schema 'main' has no qualifier
    _, source, schema, *table = table_id.split(":")[1].split(".")  # noqa: F841
    parts = table_id.split(":")[1].split(".")
    source, schema, table = parts[0], parts[1], ".".join(parts[2:])
    return table if schema == "main" else f"{schema}.{table}"


@tool
def list_sources() -> str:
    """List every data source in the semantic layer with its kind (sql or api).

    Returns a JSON array of {name, platform, kind}. Use this first to see what
    data exists before deciding how to answer a question."""
    rows = driver().execute_query(
        "MATCH (d:Database) RETURN d.name AS name, d.platform AS platform ORDER BY name",
        database_=settings.neo4j_database,
    ).records
    out = []
    for r in rows:
        platform = (r["platform"] or "").upper()
        out.append({
            "name": r["name"],
            "platform": platform,
            "kind": "sql" if platform in _SQL_PLATFORMS else "api",
        })
    return json.dumps(out)


@tool
def get_table_schema(table_id: str) -> str:
    """Get columns, types, keys, and the physical SQL reference for a table id.

    table_id looks like 'table:sales_pg.sales.order_line'. Returns JSON with
    source, sql_reference (use this in SQL), columns[], and foreign-key targets."""
    records = driver().execute_query(
        """
        MATCH (t:Table {id: $tid})-[:HAS_COLUMN]->(c:Column)
        OPTIONAL MATCH (c)-[:REFERENCES]->(rc:Column)
        RETURN c.name AS name, c.type AS type, c.is_primary_key AS pk,
               c.is_foreign_key AS fk, rc.id AS references
        ORDER BY name
        """,
        tid=table_id, database_=settings.neo4j_database,
    ).records
    if not records:
        return json.dumps({"error": f"table not found: {table_id}"})
    source = table_id.split(":")[1].split(".")[0]
    columns = [
        {"name": r["name"], "type": r["type"], "is_primary_key": r["pk"],
         "is_foreign_key": r["fk"], "references": r["references"]}
        for r in records
    ]
    return json.dumps({
        "table_id": table_id, "source": source,
        "sql_reference": _sql_reference(table_id), "columns": columns,
    })
```

- [ ] **Step 4: Run to verify it passes** (uses the `ingested_graph` fixture; first run builds the graph ~10s). Expected 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/agent/graph_tools.py backend/tests/test_agent_graph_tools.py
git commit -m "feat(agent): list_sources + get_table_schema graph tools"
```

---

## Task 3: `get_join_path` — deep-join discovery via graph traversal

**Files:**
- Modify: `backend/semantic_layer/agent/graph_tools.py`
- Test: `backend/tests/test_agent_join_path.py`

- [ ] **Step 1: Write the failing test** `backend/tests/test_agent_join_path.py`

```python
import json

import pytest

from semantic_layer.agent.graph_tools import get_join_path


@pytest.mark.neo4j
def test_join_path_segment_to_region_is_deep(ingested_graph):
    # segment and region are far apart in the sales schema -> 6+ table chain.
    result = json.loads(get_join_path.invoke({
        "table_a_id": "table:sales_pg.sales.segment",
        "table_b_id": "table:sales_pg.sales.region",
    }))
    assert result["found"] is True
    tables = result["tables"]
    assert tables[0] == "table:sales_pg.sales.segment"
    assert tables[-1] == "table:sales_pg.sales.region"
    assert len(tables) >= 6          # deep join path
    # each hop carries the join columns
    assert all("on" in hop for hop in result["joins"])


@pytest.mark.neo4j
def test_join_path_none_when_disconnected(ingested_graph):
    result = json.loads(get_join_path.invoke({
        "table_a_id": "table:sales_pg.sales.segment",
        "table_b_id": "table:financials.main.stock_price",
    }))
    assert result["found"] is False
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Append `get_join_path` to `graph_tools.py`**

```python
@tool
def get_join_path(table_a_id: str, table_b_id: str) -> str:
    """Find the shortest foreign-key join path between two tables (by id).

    Traverses REFERENCES edges in the graph and returns the ordered chain of
    tables plus the column pairs to JOIN on. Use this to build correct multi-table
    SQL — especially deep joins across many tables. Returns {found, tables, joins}."""
    records = driver().execute_query(
        """
        MATCH (ta:Table {id: $a})-[:HAS_COLUMN]->(ca:Column),
              (tb:Table {id: $b})-[:HAS_COLUMN]->(cb:Column)
        MATCH p = shortestPath((ca)-[:REFERENCES*..12]-(cb))
        RETURN [n IN nodes(p) | n.id] AS col_ids
        ORDER BY length(p) LIMIT 1
        """,
        a=table_a_id, b=table_b_id, database_=settings.neo4j_database,
    ).records
    if not records:
        return json.dumps({"found": False, "tables": [], "joins": []})
    col_ids = records[0]["col_ids"]
    # col id: col:{source}.{schema}.{table}.{column} -> table id is the prefix.
    def to_table(cid: str) -> str:
        parts = cid.split(":")[1].split(".")
        return "table:" + ".".join(parts[:-1])
    tables, joins = [], []
    for i, cid in enumerate(col_ids):
        tid = to_table(cid)
        if not tables or tables[-1] != tid:
            tables.append(tid)
        if i > 0:
            joins.append({"on": [col_ids[i - 1], cid]})
    return json.dumps({"found": True, "tables": tables, "joins": joins})
```

- [ ] **Step 4: Run to verify it passes.** Expected 2 passed. (If the segment→region path is shorter than 6 because the FK chain differs, inspect the actual `tables` and adjust the assertion to the real depth — but with the Plan 1 schema the chain is segment→product_line→product→order_line→sales_order→customer→country→region, well over 6.)

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/agent/graph_tools.py backend/tests/test_agent_join_path.py
git commit -m "feat(agent): get_join_path discovers deep FK join chains via graph traversal"
```

---

## Task 4: `search_catalog` — keyword + business-term routing

**Files:**
- Modify: `backend/semantic_layer/agent/graph_tools.py`
- Test: `backend/tests/test_agent_catalog.py`

- [ ] **Step 1: Write the failing test** `backend/tests/test_agent_catalog.py`

```python
import json

import pytest

from semantic_layer.agent.graph_tools import search_catalog


@pytest.mark.neo4j
def test_search_catalog_finds_revenue_columns(ingested_graph):
    hits = json.loads(search_catalog.invoke({"query": "revenue amount"}))
    assert len(hits) > 0
    # at least one hit should be a column on a sales table
    assert any(h["kind"] == "column" and "sales_pg" in h["table_id"] for h in hits)


@pytest.mark.neo4j
def test_search_catalog_matches_table_names(ingested_graph):
    hits = json.loads(search_catalog.invoke({"query": "ticket"}))
    assert any("itsm" in h.get("table_id", "") or "itsm" in h.get("id", "") for h in hits)
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Append `search_catalog` to `graph_tools.py`**

```python
@tool
def search_catalog(query: str, limit: int = 20) -> str:
    """Search the catalog for tables, columns, and business terms matching a query.

    Case-insensitive keyword match over names/descriptions across all sources
    (databases and APIs). Returns ranked JSON hits with their source and table so
    you can pick where to get the data. Start here to route a question."""
    terms = [t for t in query.lower().split() if len(t) > 2]
    if not terms:
        terms = [query.lower()]
    records = driver().execute_query(
        """
        UNWIND $terms AS term
        MATCH (c:Column)<-[:HAS_COLUMN]-(t:Table)
        WHERE toLower(c.name) CONTAINS term
        WITH c, t, count(*) AS score
        RETURN 'column' AS kind, c.id AS id, c.name AS name,
               t.id AS table_id, score ORDER BY score DESC LIMIT $limit
        """,
        terms=terms, limit=limit, database_=settings.neo4j_database,
    ).records
    table_hits = driver().execute_query(
        """
        UNWIND $terms AS term
        MATCH (t:Table) WHERE toLower(t.name) CONTAINS term
        WITH t, count(*) AS score
        RETURN 'table' AS kind, t.id AS id, t.name AS name,
               t.id AS table_id, score ORDER BY score DESC LIMIT $limit
        """,
        terms=terms, limit=limit, database_=settings.neo4j_database,
    ).records
    term_hits = driver().execute_query(
        """
        UNWIND $terms AS term
        MATCH (col:Column)-[:TAGGED_WITH]->(bt:BusinessTerm)
        WHERE toLower(bt.name) CONTAINS term OR toLower(coalesce(bt.description,'')) CONTAINS term
        RETURN DISTINCT 'business_term' AS kind, bt.id AS id, bt.name AS name,
               col.id AS table_id, 1 AS score LIMIT $limit
        """,
        terms=terms, limit=limit, database_=settings.neo4j_database,
    ).records
    hits = [dict(r) for r in (list(records) + list(table_hits) + list(term_hits))]
    return json.dumps(hits[:limit])
```

- [ ] **Step 4: Run to verify it passes.** Expected 2 passed. (Business-term hits only appear after a `with_llm=True` ingest; the tests assert on column/table hits which exist with the metadata-only graph.)

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/agent/graph_tools.py backend/tests/test_agent_catalog.py
git commit -m "feat(agent): search_catalog keyword + business-term routing tool"
```

---

## Task 5: `search_documents` — chunk vector search (RAG)

**Files:**
- Create: `backend/semantic_layer/agent/doc_tools.py`
- Test: `backend/tests/test_agent_doc_tools.py`

- [ ] **Step 1: Write the failing test** `backend/tests/test_agent_doc_tools.py`

```python
import json

import pytest

from semantic_layer.config import settings
from semantic_layer.graph.client import reset_graph
from semantic_layer.agent.doc_tools import search_documents


@pytest.mark.neo4j
@pytest.mark.openai
def test_search_documents_returns_relevant_chunks(neo4j_driver, require_openai):
    from semantic_layer.ingest.doc_loader import load_document
    from semantic_layer.ingest.embeddings import embed_chunks
    reset_graph(neo4j_driver)
    load_document(neo4j_driver, {
        "doc_id": "doc:t", "title": "t", "path": "/tmp/t.pdf", "num_pages": 1,
        "chunks": [
            {"chunk_id": "doc:t:chunk:0", "doc_id": "doc:t", "ordinal": 0,
             "text": "NVIDIA Data Center revenue grew on Blackwell demand."},
            {"chunk_id": "doc:t:chunk:1", "doc_id": "doc:t", "ordinal": 1,
             "text": "Gaming GPUs shipped to retail partners."},
        ],
    })
    embed_chunks(neo4j_driver)
    hits = json.loads(search_documents.invoke({"query": "data center revenue blackwell"}))
    assert len(hits) > 0
    assert hits[0]["doc_id"] == "doc:t"
    assert "revenue" in hits[0]["text"].lower()
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `backend/semantic_layer/agent/doc_tools.py`**

```python
"""Document retrieval tool: vector search over chunk embeddings."""

import json

from langchain_core.tools import tool

from semantic_layer.agent.driver import driver
from semantic_layer.config import settings
from semantic_layer.ingest.llm import get_openai_client


@tool
def search_documents(query: str, k: int = 5) -> str:
    """Search the ingested documents (NVIDIA press releases) for relevant passages.

    Embeds the query and runs vector search over document chunks. Returns the top-k
    passages with their document id and similarity score, for citing in answers."""
    vec = get_openai_client().embeddings.create(
        model=settings.embedding_model, input=[query],
        dimensions=settings.embedding_dimensions,
    ).data[0].embedding
    records = driver().execute_query(
        """
        CALL db.index.vector.queryNodes('chunk_embeddings', $k, $vec)
        YIELD node, score
        RETURN node.id AS chunk_id, node.doc_id AS doc_id,
               node.text AS text, score ORDER BY score DESC
        """,
        k=k, vec=vec, database_=settings.neo4j_database,
    ).records
    return json.dumps([
        {"chunk_id": r["chunk_id"], "doc_id": r["doc_id"],
         "text": r["text"], "score": r["score"]}
        for r in records
    ])
```

- [ ] **Step 4: Run to verify it passes** (Neo4j + OpenAI). Expected 1 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/agent/doc_tools.py backend/tests/test_agent_doc_tools.py
git commit -m "feat(agent): search_documents chunk vector-search tool"
```

---

## Task 6: `run_sql` — read-only execution with per-source routing

**Files:**
- Create: `backend/semantic_layer/agent/sql_tools.py`
- Test: `backend/tests/test_agent_sql_tools.py`

- [ ] **Step 1: Write the failing test** `backend/tests/test_agent_sql_tools.py`

```python
import json

import pytest

from semantic_layer.agent.sql_tools import run_sql


@pytest.mark.postgres
def test_run_sql_postgres_deep_join(postgres_dsn):
    sql = """
    SELECT s.name AS segment, SUM(ol.amount) AS revenue
    FROM sales.order_line ol
    JOIN sales.product p ON p.product_id = ol.product_id
    JOIN sales.product_line pl ON pl.product_line_id = p.product_line_id
    JOIN sales.segment s ON s.segment_id = pl.segment_id
    GROUP BY s.name ORDER BY revenue DESC
    """
    out = json.loads(run_sql.invoke({"source": "sales_pg", "sql": sql}))
    assert "columns" in out and "rows" in out
    assert any(r[0] == "Data Center" for r in out["rows"])


def test_run_sql_rejects_writes():
    out = json.loads(run_sql.invoke({"source": "sales_pg", "sql": "DELETE FROM sales.region"}))
    assert "error" in out


def test_run_sql_sqlite(tmp_path):
    from data.seed_sqlite import seed_all
    import semantic_layer.agent.sql_tools as st
    seed_all(out_dir=str(tmp_path))
    # point the sqlite dir at tmp for this test
    out = json.loads(st._run("financials", "SELECT COUNT(*) FROM income_statement", base_dir=str(tmp_path)))
    assert out["rows"][0][0] == 8
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `backend/semantic_layer/agent/sql_tools.py`**

```python
"""Read-only SQL execution tool with per-source engine routing."""

import json
import re
import sqlite3
from pathlib import Path

import psycopg
from langchain_core.tools import tool

from semantic_layer.config import settings

_READONLY = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_SQLITE_SOURCES = {"financials", "org"}


def _run(source: str, sql: str, base_dir: str | None = None) -> str:
    if not _READONLY.match(sql or ""):
        return json.dumps({"error": "only read-only SELECT/WITH queries are allowed"})
    limit = settings.agent_max_rows
    try:
        if source == "sales_pg":
            with psycopg.connect(settings.postgres_dsn) as conn, conn.cursor() as cur:
                cur.execute(sql)
                cols = [d.name for d in cur.description]
                rows = cur.fetchmany(limit)
        elif source in _SQLITE_SOURCES:
            path = Path(base_dir or settings.sqlite_dir) / f"{source}.db"
            con = sqlite3.connect(path)
            try:
                cur = con.execute(sql)
                cols = [d[0] for d in cur.description]
                rows = cur.fetchmany(limit)
            finally:
                con.close()
        else:
            return json.dumps({"error": f"unknown sql source '{source}'"})
    except Exception as exc:  # noqa: BLE001 — surface SQL errors back to the agent for self-repair
        return json.dumps({"error": str(exc)})
    return json.dumps({"columns": cols, "rows": [list(r) for r in rows]}, default=str)


@tool
def run_sql(source: str, sql: str) -> str:
    """Run a read-only SQL query against a structured source and return rows as JSON.

    source is one of 'sales_pg' (Postgres, tables under schema 'sales'),
    'financials', or 'org' (SQLite, unqualified table names). Only SELECT/WITH is
    allowed. On a SQL error the error text is returned so you can correct the query."""
    return _run(source, sql)
```

- [ ] **Step 4: Run to verify it passes** (Postgres seeded). Expected 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/agent/sql_tools.py backend/tests/test_agent_sql_tools.py
git commit -m "feat(agent): read-only run_sql tool with postgres/sqlite routing"
```

---

## Task 7: `call_api` — REST execution against the mock APIs

**Files:**
- Create: `backend/semantic_layer/agent/api_tools.py`
- Test: `backend/tests/test_agent_api_tools.py`

- [ ] **Step 1: Write the failing test** `backend/tests/test_agent_api_tools.py`

```python
import json

from semantic_layer.agent.api_tools import call_api


def test_call_api_lists_tickets():
    out = json.loads(call_api.invoke({"source": "itsm", "path": "/tickets", "params": {"severity": "Sev1"}}))
    assert out["status"] == 200
    assert isinstance(out["data"], list)
    assert all(t["severity"] == "Sev1" for t in out["data"])


def test_call_api_unknown_source():
    out = json.loads(call_api.invoke({"source": "nope", "path": "/x", "params": {}}))
    assert out["status"] >= 400
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `backend/semantic_layer/agent/api_tools.py`**

```python
"""Tool to call the mock enterprise REST APIs (in-process via TestClient)."""

import json

from fastapi.testclient import TestClient
from langchain_core.tools import tool

from semantic_layer.apis.app import app

_client = TestClient(app)
_SOURCES = {"crm", "itsm", "partner", "dgx"}


@tool
def call_api(source: str, path: str, params: dict | None = None) -> str:
    """Call a mock enterprise API and return its JSON.

    source is one of crm, itsm, partner, dgx. path is the endpoint under that API
    (e.g. '/tickets', '/accounts', '/inventory', '/usage'). params is an optional
    dict of query filters. Returns {status, data}. Use get_table_schema / the API's
    virtual tables to learn the available endpoints and fields."""
    if source not in _SOURCES:
        return json.dumps({"status": 404, "error": f"unknown api source '{source}'"})
    resp = _client.get(f"/{source}{path}", params=params or {})
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001
        body = resp.text
    return json.dumps({"status": resp.status_code, "data": body})
```

- [ ] **Step 4: Run to verify it passes.** Expected 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/agent/api_tools.py backend/tests/test_agent_api_tools.py
git commit -m "feat(agent): call_api tool for the mock enterprise APIs"
```

---

## Task 8: Agent assembly — subagents, orchestrator, `ask()`, golden questions

**Files:**
- Create: `backend/semantic_layer/agent/build.py`
- Test: `backend/tests/test_agent_end_to_end.py`

- [ ] **Step 1: Implement `backend/semantic_layer/agent/build.py`**

```python
"""Assemble the deepagents orchestrator with sql/api/doc subagents."""

from deepagents import create_deep_agent

from semantic_layer.config import settings
from semantic_layer.ingest.llm import get_chat_model  # also ensures OPENAI_API_KEY in env
from semantic_layer.agent.graph_tools import (
    list_sources, get_table_schema, get_join_path, search_catalog,
)
from semantic_layer.agent.doc_tools import search_documents
from semantic_layer.agent.sql_tools import run_sql
from semantic_layer.agent.api_tools import call_api

_ORCHESTRATOR_PROMPT = """You answer questions over an NVIDIA enterprise semantic layer
that unifies SQL databases, REST APIs, and documents. Workflow:
1. Use search_catalog and list_sources to find which sources/tables are relevant.
2. For structured data, use get_table_schema and get_join_path to plan the query,
   then delegate to the 'sql' subagent with the exact tables, join path, and sql_reference.
3. For enterprise-system data (CRM, support tickets, partner inventory, DGX usage),
   delegate to the 'api' subagent.
4. For narrative/press-release questions, delegate to the 'doc' subagent.
5. A question may need several subagents; combine their results.
Always state which source(s) the answer came from. Be concise and cite documents by id."""

_SQL_PROMPT = """You are a SQL expert. You are given the relevant tables, their
sql_reference values, and a join path. Write ONE read-only SELECT and run it with
run_sql(source, sql). For 'sales_pg', tables live under schema 'sales' (use the
sql_reference, e.g. sales.order_line). Use the provided join path to JOIN correctly,
including deep multi-table joins. If run_sql returns an error, fix the SQL and retry
once. Report the rows and the SQL you ran."""

_API_PROMPT = """You call mock enterprise REST APIs with call_api(source, path, params).
Sources: crm (/accounts,/contacts,/opportunities), itsm (/tickets,/rma),
partner (/partners,/inventory), dgx (/usage). Pick the endpoint and query params that
answer the question, call it, and summarize the JSON. account_id == the sales customer id."""

_DOC_PROMPT = """You answer from NVIDIA documents using search_documents(query). Retrieve
passages, then answer ONLY from them, quoting the most relevant sentence and citing the
document id. If nothing relevant is found, say so."""


def build_agent():
    model = get_chat_model()  # gpt-5.4-mini; also sets OPENAI_API_KEY in env
    subagents = [
        {"name": "sql", "description": "Runs read-only SQL over the Postgres/SQLite sources.",
         "system_prompt": _SQL_PROMPT, "tools": [get_table_schema, get_join_path, run_sql],
         "model": settings.llm_model},
        {"name": "api", "description": "Calls the CRM/ITSM/partner/DGX mock REST APIs.",
         "system_prompt": _API_PROMPT, "tools": [get_table_schema, call_api],
         "model": settings.llm_model},
        {"name": "doc", "description": "Answers from the NVIDIA documents via vector search.",
         "system_prompt": _DOC_PROMPT, "tools": [search_documents], "model": settings.llm_model},
    ]
    return create_deep_agent(
        model=model,
        tools=[list_sources, search_catalog, get_table_schema, get_join_path],
        system_prompt=_ORCHESTRATOR_PROMPT,
        subagents=subagents,
    )


def ask(question: str) -> str:
    agent = build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content
```

- [ ] **Step 2: Write the failing test** `backend/tests/test_agent_end_to_end.py`

```python
import pytest


@pytest.fixture(scope="module")
def agent_graph(ingested_graph):
    # ensure chunk embeddings exist so the doc subagent works
    from semantic_layer.ingest.embeddings import embed_chunks
    embed_chunks(ingested_graph)
    return ingested_graph


@pytest.mark.neo4j
@pytest.mark.postgres
@pytest.mark.openai
def test_structured_deep_join_question(agent_graph, require_openai):
    from semantic_layer.agent.build import ask
    answer = ask("What is total Data Center revenue, and which segment has the most revenue?")
    assert "Data Center" in answer


@pytest.mark.neo4j
@pytest.mark.postgres
@pytest.mark.openai
def test_api_question(agent_graph, require_openai):
    from semantic_layer.agent.build import ask
    answer = ask("How many open support tickets are there? Use the support system.")
    assert any(ch.isdigit() for ch in answer)


@pytest.mark.neo4j
@pytest.mark.postgres
@pytest.mark.openai
def test_document_question(agent_graph, require_openai):
    from semantic_layer.agent.build import ask
    answer = ask("According to the press releases, what drove Data Center growth?")
    assert len(answer) > 0
```

- [ ] **Step 3: Run** `cd backend && ./.venv/bin/python -m pytest tests/test_agent_end_to_end.py -v` (Neo4j + Postgres + OpenAI; each question drives the full multi-agent loop, so this is slow — allow a minute). Expected 3 passed. These are LLM-driven; if a phrasing assertion is too strict for the model's output, loosen the assertion to a robust property (e.g. the answer mentions a number / a source) but do NOT remove the test.

- [ ] **Step 4: Commit**

```bash
git add backend/semantic_layer/agent/build.py backend/tests/test_agent_end_to_end.py
git commit -m "feat(agent): deepagents orchestrator + sql/api/doc subagents + golden questions"
```

---

## Task 9: CLI, `make ask`, README, full suite

**Files:**
- Create: `backend/semantic_layer/agent/cli.py`
- Modify: `Makefile`
- Modify: `backend/README.md`

- [ ] **Step 1: Implement `backend/semantic_layer/agent/cli.py`**

```python
"""CLI: ask the semantic-layer agent a question.

Usage: python -m semantic_layer.agent.cli "your question"
"""

import sys

from semantic_layer.agent.build import ask


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python -m semantic_layer.agent.cli "question"')
        raise SystemExit(2)
    print(ask(" ".join(sys.argv[1:])))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add an `ask` target to the repo-root `Makefile`** (add to `.PHONY`; recipe TAB-indented). `q` is passed on the command line: `make ask q="..."`.

```makefile
ask:
	cd backend && python -m semantic_layer.agent.cli "$(q)"
```

Verify: `make -n ask q="hello"` prints the command.

- [ ] **Step 3: Append an Agent section to `backend/README.md`** documenting prerequisites (`make up`, `make seed`, `make ingest`, `OPENAI_API_KEY`), `make ask q="..."`, the orchestrator + three subagents, and example questions (a deep-join structured one, an API one, a document one, and a cross-source one).

- [ ] **Step 4: Run the full suite** `cd backend && ./.venv/bin/python -m pytest -q` and paste the summary; nothing should fail (agent end-to-end tests run live or skip without services/key).

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/agent/cli.py Makefile backend/README.md
git commit -m "feat(agent): ask CLI, make ask target, and agent docs"
```

---

## Self-Review

**Spec coverage (Plan 4 scope):** deepagents orchestrator on gpt-5.4-mini via init_chat_model (Task 8) ✓; semantic tools — `search_catalog` (Task 4), `get_table_schema` (Task 2), `get_join_path` graph traversal for deep joins (Task 3), `list_sources` (Task 2), `search_documents` vector RAG (Task 5) ✓; three subagents — sql (grounded text-to-SQL, read-only, self-repair) (Tasks 6, 8), api (Tasks 7, 8), doc (Tasks 5, 8) ✓; synthesis with provenance (orchestrator prompt, Task 8) ✓; cross-source golden questions (Task 8) ✓; `make ask` entrypoint (Task 9) ✓. The web UI is Plan 5.

**External-API honesty:** `create_deep_agent` is called with the verified signature; subagents are plain `SubAgent` dicts with the verified required keys; tools use `@tool`; `db.index.vector.queryNodes('chunk_embeddings', ...)` uses the index name WE created in Plan 3. No guessed APIs.

**Grounding, not blind text-to-SQL:** the sql subagent receives the tables, `sql_reference`, and `get_join_path` result from the orchestrator before writing SQL — the join path comes from the graph, which is the whole point of the semantic layer for deep joins.

**Marker discipline:** graph tools tests use a session `ingested_graph` fixture (neo4j+postgres, builds the graph once, no LLM); doc/agent tests add `openai`; everything skips cleanly without the relevant service/key. `run_sql` is read-only (SELECT/WITH guard) and caps rows at `settings.agent_max_rows`.

**Type/name consistency:** `Table.id`/`Column.id` formats parsed identically in `_sql_reference`, `get_table_schema`, and `get_join_path`. Tool names (`list_sources`, `search_catalog`, `get_table_schema`, `get_join_path`, `search_documents`, `run_sql`, `call_api`) are defined once and imported unchanged into `build.py`. Source names (`sales_pg`, `financials`, `org`, `crm`, `itsm`, `partner`, `dgx`) are consistent across `run_sql` routing, `call_api` `_SOURCES`, and the graph `Database.name`. `settings.llm_model` / `embedding_model` / `embedding_dimensions` / `agent_max_rows` referenced consistently.

**Scope check:** one coherent subsystem (the agent over the existing graph + sources). 9 tasks, each independently testable. The UI is deferred to Plan 5.
```
