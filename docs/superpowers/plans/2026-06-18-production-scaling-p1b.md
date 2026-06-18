# Production-Scaling P1b Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close out the P1 follow-ups — the three Minor cleanups from the P1 final review, a top-k join-path capability, and an offline LLM-judge evaluation harness that makes schema-routing tuning and answer-quality regression measurable.

**Architecture:** Three independent features. (A) Minor hardening: a once-per-process Postgres pool open + FastAPI lifespan warm-up, a multi-worker connection-sizing doc note, and a defensive guard on the D1 fact-rank Cypher. (B) `k_shortest_join_paths` returns the top-k join paths ranked by observed-join weight (alternatives the planner/agent can fall back to), built on the same `allShortestPaths` traversal `get_join_path` already uses. (C) An eval harness (`semantic_layer/eval/`): a golden-question evalset, an LLM judge scoring answers 1–4 against expectations, and a runner+report, defaulting to the deterministic `controller.answer_stream` path.

**Tech Stack:** Python ≥3.11, Neo4j (`neo4j` driver), psycopg3 + psycopg_pool, FastAPI, LangChain (`openai:gpt-5.4-mini`), pydantic, pytest.

## Global Constraints

- **Python:** `requires-python = ">=3.11"` (`backend/pyproject.toml`). No new runtime dependency without adding it to `pyproject.toml`.
- **Models from config only:** chat via `semantic_layer.ingest.llm.get_chat_model(<settings id>)`; for the judge use `settings.synthesis_model_resolved`. Never hardcode model ids.
- **Tests run from `backend/` with the project venv:** `backend/.venv/bin/python -m pytest`. The Makefile's bare `python` is NOT on PATH (use the venv binary, or `source backend/.venv/bin/activate`). `pythonpath = ["."]`.
- **Test markers (`backend/pyproject.toml`):** `neo4j`, `postgres`, `openai`. Marker-gated tests MUST take the matching fixture (`neo4j_driver`/`ingested_graph`, `postgres_dsn`, `require_openai`) so they SKIP when the service/key is absent. Pure-logic tests take no fixture.
- **LLM stubbing convention (from `tests/test_legs.py`):** monkeypatch the module-level `get_chat_model` with a fake whose `.with_structured_output(...)` returns a `_FakeStructured` with a fixed `.invoke()` value. Never call a real model in a non-`openai` test.
- **Graph writes use `MERGE`**; the new Cypher in this plan is read-only.
- **⚠️ Running the full backend pytest suite wipes `Chunk.embedding`** (neo4j tests reset the shared graph). After a DB-touching run, restore with `source backend/.venv/bin/activate && make ingest` (or `backend/.venv/bin/python -m semantic_layer.ingest.pipeline`). Pure-logic tasks don't trigger this.
- **TDD + frequent commits:** every task is test-first and ends with a commit.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `backend/semantic_layer/agent/pg_pool.py` (modify) | `ensure_pool_open()` once-per-process | A1 |
| `backend/semantic_layer/agent/sql_tools.py` (modify) | call `ensure_pool_open()` not per-call `pool.open()` | A1 |
| `backend/semantic_layer/web/app.py` (modify) | FastAPI lifespan: warm pool on startup, close on shutdown | A1 |
| `backend/semantic_layer/config.py` (modify) | multi-worker sizing doc note on pool/concurrency settings | A1 |
| `backend/semantic_layer/agent/routing.py` (modify) | `WHERE t2 <> t` guard in `_FACT_RANK_CYPHER` | A2 |
| `backend/semantic_layer/agent/graph_tools.py` (modify) | `k_shortest_join_paths(a, b, k)` | B1 |
| `backend/semantic_layer/eval/__init__.py` (create) | package marker | C1 |
| `backend/semantic_layer/eval/evalset.py` (create) | evalset loader | C1 |
| `backend/data/eval/golden_questions.json` (create) | golden questions + expectations | C1 |
| `backend/semantic_layer/eval/judge.py` (create) | LLM judge (1–4) | C2 |
| `backend/semantic_layer/eval/run.py` (create) | runner + report + CLI | C3 |
| `backend/tests/test_pg_pool_lifecycle.py` (create) | ensure_pool_open + lifespan | A1 |
| `backend/tests/test_routing_fact_table.py` (modify) | depth-2 self-FK guard | A2 |
| `backend/tests/test_k_shortest_paths.py` (create) | top-k join paths | B1 |
| `backend/tests/test_eval_harness.py` (create) | evalset + judge + runner | C1, C2, C3 |

Features A, B, C are mutually independent. Within A, A1 and A2 touch different files and are independent. Within C, do C1 → C2 → C3 in order.

---

# Feature A — P1 Minor Cleanups

The three Minor findings from the P1 whole-branch review.

---

### Task A1: Once-per-process pool open + lifespan warm-up + sizing doc note

**Files:**
- Modify: `backend/semantic_layer/agent/pg_pool.py`
- Modify: `backend/semantic_layer/agent/sql_tools.py`
- Modify: `backend/semantic_layer/web/app.py`
- Modify: `backend/semantic_layer/config.py`
- Test: `backend/tests/test_pg_pool_lifecycle.py`

**Interfaces:**
- Produces: `ensure_pool_open() -> None` in `pg_pool.py` — opens the cached pool at most once per process (module guard), idempotent and safe to call from any caller (web, CLI, ingest, tests). `sql_tools._run` calls it instead of `pool.open()`. `web/app.py` gets a FastAPI lifespan that calls `ensure_pool_open()` on startup and `get_pool().close()` on shutdown.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_pg_pool_lifecycle.py
from semantic_layer.agent import pg_pool


class _FakePool:
    def __init__(self):
        self.open_calls = 0
        self.close_calls = 0

    def open(self):
        self.open_calls += 1

    def close(self):
        self.close_calls += 1


def test_ensure_pool_open_opens_at_most_once(monkeypatch):
    fake = _FakePool()
    monkeypatch.setattr(pg_pool, "get_pool", lambda: fake)
    monkeypatch.setattr(pg_pool, "_pool_opened", False, raising=False)
    pg_pool.ensure_pool_open()
    pg_pool.ensure_pool_open()
    pg_pool.ensure_pool_open()
    assert fake.open_calls == 1   # guarded: opened once despite 3 calls
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_pg_pool_lifecycle.py -v`
Expected: FAIL with `AttributeError: module 'semantic_layer.agent.pg_pool' has no attribute 'ensure_pool_open'`.

- [ ] **Step 3: Implement `ensure_pool_open`**

In `backend/semantic_layer/agent/pg_pool.py`, add a module-level guard and function below `get_pool`:

```python
_pool_opened = False


def ensure_pool_open() -> None:
    """Open the cached pool exactly once per process (idempotent, thread-safe enough
    for our use — a redundant open is harmless). Callable from web startup, the CLI,
    ingest, or sql_tools; avoids re-calling pool.open() on every query."""
    global _pool_opened
    if not _pool_opened:
        get_pool().open()
        _pool_opened = True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_pg_pool_lifecycle.py::test_ensure_pool_open_opens_at_most_once -v`
Expected: PASS.

- [ ] **Step 5: Use it in `sql_tools._run`**

In `backend/semantic_layer/agent/sql_tools.py`, change the import:

```python
from semantic_layer.agent.pg_pool import get_pool, ensure_pool_open
```

and in the `sales_pg` branch replace `pool.open()` with `ensure_pool_open()`:

```python
        if source == "sales_pg":
            ensure_pool_open()
            with get_pool().connection() as conn, conn.cursor() as cur:
                cur.execute(sql) if params is None else cur.execute(sql, params)
                cols = [d.name for d in cur.description]
                rows = cur.fetchmany(limit)
```

- [ ] **Step 6: Add the FastAPI lifespan (failing test first)**

Add to `backend/tests/test_pg_pool_lifecycle.py`:

```python
def test_app_lifespan_warms_and_closes_pool(monkeypatch):
    from fastapi.testclient import TestClient
    from semantic_layer.web import app as app_mod

    calls = {"open": 0, "close": 0}
    monkeypatch.setattr(app_mod, "ensure_pool_open", lambda: calls.__setitem__("open", calls["open"] + 1))

    class _P:
        def close(self):
            calls["close"] += 1

    monkeypatch.setattr(app_mod, "get_pool", lambda: _P())
    with TestClient(app_mod.app):          # context manager triggers lifespan startup/shutdown
        assert calls["open"] == 1
    assert calls["close"] == 1
```

- [ ] **Step 7: Run to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_pg_pool_lifecycle.py::test_app_lifespan_warms_and_closes_pool -v`
Expected: FAIL with `AttributeError: module 'semantic_layer.web.app' has no attribute 'ensure_pool_open'` (the name isn't imported / no lifespan yet).

- [ ] **Step 8: Implement the lifespan**

In `backend/semantic_layer/web/app.py`, add imports and a lifespan, and attach it to the app. Replace the top of the file through `app = FastAPI(...)`:

```python
"""FastAPI web API for the semantic-layer UI."""

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from semantic_layer.web.graph_api import get_sources, get_schema_graph
from semantic_layer.web.events import stream_chat_events
from semantic_layer.agent.pg_pool import ensure_pool_open, get_pool


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_pool_open()        # warm the Postgres pool at startup
    yield
    get_pool().close()        # release pooled connections on shutdown


app = FastAPI(title="NeoCarta-Local Web API", lifespan=lifespan)
```

(Leave the rest of `app.py` — CORS, routes — unchanged.)

- [ ] **Step 9: Add the multi-worker sizing doc note**

In `backend/semantic_layer/config.py`, replace the pool-settings comment block so it warns about per-worker multiplication. Change:

```python
    # Postgres connection pool (replaces per-call psycopg.connect in sql_tools).
    pg_pool_min_size: int = 1
    pg_pool_max_size: int = 8
```

to:

```python
    # Postgres connection pool (replaces per-call psycopg.connect in sql_tools).
    # NOTE (multi-worker): the pool and the answer-concurrency gate are PER PROCESS.
    # Under N uvicorn/gunicorn workers the real Postgres connection ceiling is
    # N * pg_pool_max_size — size it against Postgres `max_connections`, and treat
    # max_concurrent_answers as a per-worker limit (cluster ceiling = N * value).
    pg_pool_min_size: int = 1
    pg_pool_max_size: int = 8
```

- [ ] **Step 10: Run the full lifecycle test + existing pool/web tests**

Run: `cd backend && .venv/bin/python -m pytest tests/test_pg_pool_lifecycle.py tests/test_pg_pool.py tests/test_web_chat_events.py tests/test_api_app.py -v`
Expected: new tests PASS; existing pool/web tests stay green (the postgres-marked `_run` test still uses the pool; the lifespan only runs under a TestClient context manager).

- [ ] **Step 11: Commit**

```bash
git add backend/semantic_layer/agent/pg_pool.py backend/semantic_layer/agent/sql_tools.py backend/semantic_layer/web/app.py backend/semantic_layer/config.py backend/tests/test_pg_pool_lifecycle.py
git commit -m "fix(pg): open pool once + lifespan warm-up; document per-worker sizing"
```

---

### Task A2: Defensive `WHERE t2 <> t` guard on the fact-rank depth-2 reach

**Files:**
- Modify: `backend/semantic_layer/agent/routing.py`
- Test: `backend/tests/test_routing_fact_table.py`

**Interfaces:**
- Consumes/Produces: `_FACT_RANK_CYPHER` — the depth-2 `OPTIONAL MATCH` excludes paths that loop back to the start table (`t2 <> t`), so a self- or mutually-referential FK cannot inflate a table's `depth2` reach. `select_fact_table`'s public behavior is unchanged on the current (acyclic) schema.

- [ ] **Step 1: Write the failing test (neo4j, builds a self-referential probe)**

```python
# append to backend/tests/test_routing_fact_table.py
@pytest.mark.neo4j
def test_select_fact_table_depth2_excludes_self_reference(ingested_graph):
    # Build two sales-schema probe tables that BOTH have 1 direct FK, where one (selfy)
    # also has a self-referential FK column that, WITHOUT the t2<>t guard, would inflate
    # its depth-2 reach and let it beat the genuine 2-hop table (reachy).
    with ingested_graph.session() as s:
        s.run(
            """
            MERGE (selfy:Table {id:'table:sales_pg.sales._probe_selfy'})
            MERGE (reachy:Table {id:'table:sales_pg.sales._probe_reachy'})
            MERGE (mid:Table {id:'table:sales_pg.sales._probe_mid'})
            MERGE (leaf:Table {id:'table:sales_pg.sales._probe_leaf'})
            // selfy: 1 FK to mid, plus a self-FK (col_s2 -> col_s_pk on the SAME table)
            MERGE (selfy)-[:HAS_COLUMN]->(cs1:Column {id:'col:_p.selfy.fk_mid'})
            MERGE (selfy)-[:HAS_COLUMN]->(cs2:Column {id:'col:_p.selfy.fk_self'})
            MERGE (selfy)-[:HAS_COLUMN]->(csp:Column {id:'col:_p.selfy.pk'})
            MERGE (cs2)-[:REFERENCES]->(csp)
            MERGE (mid)-[:HAS_COLUMN]->(cm1:Column {id:'col:_p.mid.pk'})
            MERGE (cs1)-[:REFERENCES]->(cm1)
            // reachy: 1 FK to mid; mid has its own FK to leaf -> genuine depth-2 reach = 1
            MERGE (reachy)-[:HAS_COLUMN]->(cr1:Column {id:'col:_p.reachy.fk_mid'})
            MERGE (cr1)-[:REFERENCES]->(cm1)
            MERGE (mid)-[:HAS_COLUMN]->(cm2:Column {id:'col:_p.mid.fk_leaf'})
            MERGE (leaf)-[:HAS_COLUMN]->(cl1:Column {id:'col:_p.leaf.pk'})
            MERGE (cm2)-[:REFERENCES]->(cl1)
            """
        )
    try:
        result = routing.select_fact_table([
            "table:sales_pg.sales._probe_selfy", "table:sales_pg.sales._probe_reachy"])
        # Both have 1 direct FK; reachy has genuine depth-2 reach (mid->leaf), selfy's only
        # extra "reach" is the self-loop which the guard must exclude -> reachy wins.
        assert result == "table:sales_pg.sales._probe_reachy"
    finally:
        with ingested_graph.session() as s:
            s.run(
                "MATCH (t:Table) WHERE t.id STARTS WITH 'table:sales_pg.sales._probe_' "
                "OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c) DETACH DELETE t, c")
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_routing_fact_table.py::test_select_fact_table_depth2_excludes_self_reference -v`
Expected: FAIL — without the guard, `selfy`'s self-FK adds to its depth-2 reach and it ties/beats `reachy` (assertion fails), or SKIP if Neo4j unavailable.

- [ ] **Step 3: Implement the guard**

In `backend/semantic_layer/agent/routing.py`, in `_FACT_RANK_CYPHER`, add a `WHERE t2 <> t` to the depth-2 step. Change:

```cypher
OPTIONAL MATCH (t)-[:HAS_COLUMN]->(:Column)-[:REFERENCES]->(:Column)<-[:HAS_COLUMN]-(:Table)-[:HAS_COLUMN]->(:Column)-[:REFERENCES]->(:Column)<-[:HAS_COLUMN]-(t2:Table)
WITH tid, fks, count(DISTINCT t2) AS depth2
```

to:

```cypher
OPTIONAL MATCH (t)-[:HAS_COLUMN]->(:Column)-[:REFERENCES]->(:Column)<-[:HAS_COLUMN]-(:Table)-[:HAS_COLUMN]->(:Column)-[:REFERENCES]->(:Column)<-[:HAS_COLUMN]-(t2:Table)
WHERE t2 <> t
WITH tid, fks, count(DISTINCT t2) AS depth2
```

- [ ] **Step 4: Run the new test + existing fact-table tests**

Run: `cd backend && .venv/bin/python -m pytest tests/test_routing_fact_table.py -v`
Expected: the self-reference test PASS; the existing `test_select_fact_table_picks_order_line` still selects `order_line` (the guard doesn't change acyclic-schema results). (neo4j tests SKIP if no DB.)

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/agent/routing.py backend/tests/test_routing_fact_table.py
git commit -m "fix(routing): exclude self-reference from fact-rank depth-2 reach"
```

---

# Feature B — Top-k Join Paths

**Why:** `get_join_path` returns exactly one path (shortest, highest observed-join weight). Exposing the top-k alternatives lets the planner/agent fall back when the best path's SQL fails and lets the UI show alternatives. This builds on the same bounded `allShortestPaths` traversal `get_join_path` already uses (`graph_tools.py:118-140`), so it is safe and non-explosive on the schema.

> Scope note: this returns the k best **shortest-length** paths ranked by observed-join weight — not Yen's longer-than-shortest alternatives. True longer-path k-shortest needs the Neo4j GDS library (`gds.shortestPath.yens`), which is not installed (only APOC is). That remains a further follow-up if minimal-length paths ever prove insufficient on a larger schema.

---

### Task B1: `k_shortest_join_paths`

**Files:**
- Modify: `backend/semantic_layer/agent/graph_tools.py`
- Test: `backend/tests/test_k_shortest_paths.py`

**Interfaces:**
- Consumes: `driver()`, `settings.neo4j_database`.
- Produces: `k_shortest_join_paths(table_a_id: str, table_b_id: str, k: int = 3) -> str` (a LangChain `@tool`) returning JSON `{found: bool, paths: [{tables: [...], joins: [{on:[col,col]}], observed: int}]}` — up to `k` shortest-length paths ordered by total `OBSERVED_JOIN` weight DESC. `a == b` returns one zero-hop path. No path found → `{found: false, paths: []}`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_k_shortest_paths.py
import json

import pytest

from semantic_layer.agent.graph_tools import k_shortest_join_paths


def test_same_table_returns_zero_hop_path():
    out = json.loads(k_shortest_join_paths.invoke({
        "table_a_id": "table:sales_pg.sales.order_line",
        "table_b_id": "table:sales_pg.sales.order_line"}))
    assert out["found"] is True
    assert out["paths"][0]["tables"] == ["table:sales_pg.sales.order_line"]
    assert out["paths"][0]["joins"] == []


@pytest.mark.neo4j
def test_returns_ranked_paths_between_tables(ingested_graph):
    out = json.loads(k_shortest_join_paths.invoke({
        "table_a_id": "table:sales_pg.sales.order_line",
        "table_b_id": "table:sales_pg.sales.region", "k": 3}))
    assert out["found"] is True
    assert 1 <= len(out["paths"]) <= 3
    # Each path connects the two tables and carries an observed-weight score.
    for p in out["paths"]:
        assert p["tables"][0] == "table:sales_pg.sales.order_line"
        assert p["tables"][-1] == "table:sales_pg.sales.region"
        assert isinstance(p["observed"], int)
    # Ranked by observed weight DESC.
    weights = [p["observed"] for p in out["paths"]]
    assert weights == sorted(weights, reverse=True)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_k_shortest_paths.py -v`
Expected: FAIL with `ImportError: cannot import name 'k_shortest_join_paths'`.

- [ ] **Step 3: Implement**

In `backend/semantic_layer/agent/graph_tools.py`, add (it reuses the same column-pairing logic as `get_join_path`):

```python
@tool
def k_shortest_join_paths(table_a_id: str, table_b_id: str, k: int = 3) -> str:
    """Return up to k shortest join paths between two tables, ranked by observed-join weight.

    Like get_join_path but returns ALTERNATIVES (the planner/agent can fall back to a
    different path if the top one's SQL fails). Each path is {tables, joins, observed}.
    Returns {found, paths}."""
    if table_a_id == table_b_id:
        return json.dumps({"found": True,
                           "paths": [{"tables": [table_a_id], "joins": [], "observed": 0}]})
    try:
        records = driver().execute_query(
            """
            MATCH (ta:Table {id: $a}), (tb:Table {id: $b})
            MATCH p = allShortestPaths((ta)-[:HAS_COLUMN|REFERENCES|OBSERVED_JOIN|SAME_ENTITY*1..24]-(tb))
            WITH p, reduce(w = 0, r IN relationships(p) | w + coalesce(r.observations, 0)) AS observed
            RETURN [n IN nodes(p) | head(labels(n)) + '|' + n.id] AS nodes, observed
            ORDER BY observed DESC LIMIT $k
            """,
            a=table_a_id, b=table_b_id, k=k, database_=settings.neo4j_database,
        ).records
    except Exception as exc:  # noqa: BLE001 — surface graph errors, don't crash the run
        return json.dumps({"found": False, "paths": [], "error": str(exc)})
    paths = []
    for rec in records:
        nodes = rec["nodes"]
        tables = [n.split("|", 1)[1] for n in nodes if n.startswith("Table|")]
        cols = [n.split("|", 1)[1] for n in nodes if n.startswith("Column|")]
        joins = [{"on": [cols[i], cols[i + 1]]} for i in range(0, len(cols) - 1, 2)]
        paths.append({"tables": tables, "joins": joins, "observed": rec["observed"]})
    return json.dumps({"found": bool(paths), "paths": paths})
```

- [ ] **Step 4: Run the new test + existing join-path tests**

Run: `cd backend && .venv/bin/python -m pytest tests/test_k_shortest_paths.py tests/test_agent_join_path.py -v`
Expected: new tests PASS (neo4j SKIP if no DB); `test_agent_join_path.py` (existing `get_join_path`) stays green — untouched.

- [ ] **Step 5: Verify latency on the live graph (no code change)**

If Neo4j is available, confirm the query is fast on the real schema:
Run: `cd backend && .venv/bin/python -c "import time,json; from semantic_layer.agent.graph_tools import k_shortest_join_paths as f; t=time.perf_counter(); print(len(json.loads(f.invoke({'table_a_id':'table:sales_pg.sales.order_line','table_b_id':'table:sales_pg.sales.region','k':3}))['paths'])); print(round((time.perf_counter()-t)*1000,1),'ms')"`
Expected: prints a path count (1–3) and a latency well under 100 ms. If it is slow, lower the `*1..24` hop cap to `*1..12` in the query and re-run.

- [ ] **Step 6: Commit**

```bash
git add backend/semantic_layer/agent/graph_tools.py backend/tests/test_k_shortest_paths.py
git commit -m "feat(graph): k_shortest_join_paths — top-k join paths by observed weight"
```

---

# Feature C — Offline Eval Harness

**Why:** There is no way to measure whether a change (schema-routing tuning, prompt edits, cache changes) improves or regresses answer quality. This adds a golden-question evalset, an LLM judge (1–4), and a runner so quality is a number you can track. It targets the deterministic `controller.answer_stream` path (the web/UI path optimized in P0/P1).

---

### Task C1: Evalset format + loader

**Files:**
- Create: `backend/semantic_layer/eval/__init__.py`
- Create: `backend/semantic_layer/eval/evalset.py`
- Create: `backend/data/eval/golden_questions.json`
- Test: `backend/tests/test_eval_harness.py`

**Interfaces:**
- Produces: `load_evalset(path: str | None = None) -> list[dict]` — loads the JSON evalset; each item is `{"id": str, "question": str, "expect": str}` (`expect` = a plain-language description of what a correct answer must contain). Default path is `settings`-relative `data/eval/golden_questions.json`.

- [ ] **Step 1: Create the package marker and the evalset data**

Create `backend/semantic_layer/eval/__init__.py`:

```python
"""Offline evaluation harness for the semantic-layer agent."""
```

Create `backend/data/eval/golden_questions.json` (seeded from the existing golden questions in `tests/test_agent_end_to_end.py` plus a cross-source one):

```json
[
  {"id": "deep_join_segment_revenue",
   "question": "Which business segment has the highest total revenue? Use the sales database.",
   "expect": "Names Data Center as the highest-revenue segment, sourced from the sales SQL database."},
  {"id": "api_open_tickets",
   "question": "How many open support tickets are there? Use the support system.",
   "expect": "Gives a specific integer count of open tickets from the ITSM/support API."},
  {"id": "doc_dc_growth",
   "question": "According to the press releases, what drove Data Center growth?",
   "expect": "Summarizes Data Center growth drivers and cites a press-release document id."},
  {"id": "cross_source_blackwell_emea",
   "question": "For EMEA Cloud customers, what was Blackwell Data Center revenue, and what do the press releases say about Blackwell?",
   "expect": "Reports a SQL revenue figure scoped to EMEA region + Cloud industry + Blackwell architecture + Data Center segment, AND a press-release statement about Blackwell, each with its source."}
]
```

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/test_eval_harness.py
from semantic_layer.eval.evalset import load_evalset


def test_load_evalset_default_has_expected_shape():
    items = load_evalset()
    assert len(items) >= 4
    for it in items:
        assert set(it) >= {"id", "question", "expect"}
        assert isinstance(it["question"], str) and it["question"]
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids))   # unique ids
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_eval_harness.py::test_load_evalset_default_has_expected_shape -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'semantic_layer.eval.evalset'`.

- [ ] **Step 4: Implement the loader**

Create `backend/semantic_layer/eval/evalset.py`:

```python
"""Load the golden-question evalset."""

import json
from pathlib import Path

from semantic_layer.config import settings

_DEFAULT_PATH = "data/eval/golden_questions.json"


def load_evalset(path: str | None = None) -> list[dict]:
    """Load evalset items [{id, question, expect}, ...] from JSON.

    Default path is resolved relative to the configured sqlite_dir's parent (the
    `data/` root used elsewhere), falling back to the literal default path."""
    p = Path(path) if path else Path(_DEFAULT_PATH)
    if not p.exists():
        # sqlite_dir is 'data/seed'; the eval set lives under the same data root.
        p = Path(settings.sqlite_dir).parent / "eval" / "golden_questions.json"
    return json.loads(p.read_text())
```

- [ ] **Step 5: Run to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_eval_harness.py::test_load_evalset_default_has_expected_shape -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/semantic_layer/eval/__init__.py backend/semantic_layer/eval/evalset.py backend/data/eval/golden_questions.json backend/tests/test_eval_harness.py
git commit -m "feat(eval): golden-question evalset + loader"
```

---

### Task C2: LLM judge (1–4)

**Files:**
- Create: `backend/semantic_layer/eval/judge.py`
- Test: `backend/tests/test_eval_harness.py`

**Interfaces:**
- Consumes: `get_chat_model` (`ingest/llm.py`), `settings.synthesis_model_resolved`.
- Produces: `judge_answer(question: str, answer: str, expect: str) -> dict` → `{"score": int (1-4), "reason": str}` via one structured LLM call. Score rubric: 4 = fully correct & complete, 3 = mostly correct minor gap, 2 = partially correct, 1 = wrong/missing.

- [ ] **Step 1: Write the failing test**

```python
# append to backend/tests/test_eval_harness.py
from semantic_layer.eval import judge as judge_mod
from semantic_layer.eval.judge import judge_answer


class _FakeStructured:
    def __init__(self, value):
        self._value = value

    def invoke(self, _messages):
        return self._value


class _FakeModel:
    def __init__(self, value):
        self._value = value

    def with_structured_output(self, _schema, **_kwargs):
        return _FakeStructured(self._value)


def test_judge_answer_returns_score_and_reason(monkeypatch):
    verdict = judge_mod._Verdict(score=4, reason="Names Data Center, cites sales DB.")
    monkeypatch.setattr(judge_mod, "get_chat_model", lambda model=None: _FakeModel(verdict))
    out = judge_answer("Which segment leads revenue?",
                       "Data Center leads, per the sales database.",
                       "Names Data Center as highest-revenue segment from sales SQL.")
    assert out["score"] == 4
    assert "Data Center" in out["reason"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_eval_harness.py::test_judge_answer_returns_score_and_reason -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'semantic_layer.eval.judge'`.

- [ ] **Step 3: Implement the judge**

Create `backend/semantic_layer/eval/judge.py`:

```python
"""LLM-as-judge: score an agent answer 1-4 against the expectation."""

from pydantic import BaseModel, Field

from semantic_layer.config import settings
from semantic_layer.ingest.llm import get_chat_model


class _Verdict(BaseModel):
    score: int = Field(ge=1, le=4)
    reason: str


_JUDGE_PROMPT = (
    "You are grading an answer from an enterprise data agent against a description of "
    "what a correct answer must contain. Score 1-4: 4 = fully correct AND complete; "
    "3 = mostly correct, minor omission; 2 = partially correct or missing a required "
    "part; 1 = wrong, unsupported, or non-answer. Judge only against the expectation; "
    "do not reward extra unverifiable claims. Return score and a one-sentence reason."
)


def judge_answer(question: str, answer: str, expect: str) -> dict:
    """One structured LLM call scoring `answer` 1-4 against `expect`."""
    model = get_chat_model(settings.synthesis_model_resolved).with_structured_output(_Verdict)
    v = model.invoke([
        ("system", _JUDGE_PROMPT),
        ("human", f"Question:\n{question}\n\nExpectation:\n{expect}\n\nAnswer:\n{answer}"),
    ])
    return {"score": v.score, "reason": v.reason}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_eval_harness.py::test_judge_answer_returns_score_and_reason -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/eval/judge.py backend/tests/test_eval_harness.py
git commit -m "feat(eval): LLM judge scoring answers 1-4"
```

---

### Task C3: Runner + report + CLI

**Files:**
- Create: `backend/semantic_layer/eval/run.py`
- Test: `backend/tests/test_eval_harness.py`

**Interfaces:**
- Consumes: `load_evalset` (C1), `judge_answer` (C2), `controller.answer_stream`.
- Produces:
  - `default_answer_fn(question: str) -> str` — drives `controller.answer_stream` and returns the final `answer` event's `content`.
  - `run_eval(evalset: list[dict], answer_fn=default_answer_fn, judge_fn=judge_answer) -> dict` → `{"results": [{id, score, reason}], "mean_score": float, "pass_rate": float}` where pass = score ≥ 3.
  - `main()` — CLI (`python -m semantic_layer.eval.run`) that loads the default evalset, runs it, and prints the report as JSON.

- [ ] **Step 1: Write the failing test**

```python
# append to backend/tests/test_eval_harness.py
from semantic_layer.eval import run as run_mod
from semantic_layer.eval.run import run_eval


def test_run_eval_aggregates_scores():
    evalset = [
        {"id": "a", "question": "q1", "expect": "e1"},
        {"id": "b", "question": "q2", "expect": "e2"},
        {"id": "c", "question": "q3", "expect": "e3"},
    ]
    answers = {"q1": "A1", "q2": "A2", "q3": "A3"}
    scores = {"q1": 4, "q2": 3, "q3": 1}
    report = run_eval(
        evalset,
        answer_fn=lambda q: answers[q],
        judge_fn=lambda question, answer, expect: {"score": scores[question], "reason": "r"},
    )
    assert [r["id"] for r in report["results"]] == ["a", "b", "c"]
    assert report["mean_score"] == round((4 + 3 + 1) / 3, 2)
    assert report["pass_rate"] == round(2 / 3, 2)   # a & b pass (>=3), c fails


def test_default_answer_fn_extracts_final_answer(monkeypatch):
    events = [
        {"type": "tool_result", "scope": "sql", "content": "{}"},
        {"type": "answer", "content": "the final answer", "highlight": []},
    ]
    monkeypatch.setattr(run_mod, "answer_stream", lambda q: iter(events))
    assert run_mod.default_answer_fn("anything") == "the final answer"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_eval_harness.py::test_run_eval_aggregates_scores tests/test_eval_harness.py::test_default_answer_fn_extracts_final_answer -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'semantic_layer.eval.run'`.

- [ ] **Step 3: Implement the runner**

Create `backend/semantic_layer/eval/run.py`:

```python
"""Run the golden-question evalset through the agent and score with the LLM judge."""

import json

from semantic_layer.agent.controller import answer_stream
from semantic_layer.eval.evalset import load_evalset
from semantic_layer.eval.judge import judge_answer


def default_answer_fn(question: str) -> str:
    """Drive the deterministic controller path and return the final answer text."""
    final = ""
    for event in answer_stream(question):
        if event.get("type") == "answer":
            final = event.get("content", "")
    return final


def run_eval(evalset: list[dict], answer_fn=default_answer_fn, judge_fn=judge_answer) -> dict:
    """Answer each question and judge it; return per-item results plus aggregates."""
    results = []
    for item in evalset:
        answer = answer_fn(item["question"])
        verdict = judge_fn(item["question"], answer, item["expect"])
        results.append({"id": item["id"], "score": verdict["score"], "reason": verdict["reason"]})
    n = len(results) or 1
    mean = round(sum(r["score"] for r in results) / n, 2)
    pass_rate = round(sum(1 for r in results if r["score"] >= 3) / n, 2)
    return {"results": results, "mean_score": mean, "pass_rate": pass_rate}


def main() -> None:
    report = run_eval(load_evalset())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_eval_harness.py -v`
Expected: all eval tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/eval/run.py backend/tests/test_eval_harness.py
git commit -m "feat(eval): runner + report + CLI over the controller answer path"
```

---

## Self-Review

**Spec coverage:**
- ✅ Minor #1 (pool open once + lifespan) — Task A1.
- ✅ Minor #2 (multi-worker sizing doc note) — Task A1 Step 9.
- ✅ Minor #3 (depth-2 `WHERE t2 <> t`) — Task A2.
- ✅ Weighted/k-shortest join discovery — Feature B (scoped to k-best shortest paths by observed weight; Yen's-via-GDS noted as a further follow-up).
- ✅ Offline eval harness — Feature C (evalset C1, judge C2, runner+CLI C3).

**Placeholder scan:** No TBDs; every code step has complete code; every run step states the expected result and the venv command.

**Type consistency:**
- `ensure_pool_open() -> None` consumed by `sql_tools._run` and `web/app.py` lifespan — consistent (A1).
- `k_shortest_join_paths(a, b, k) -> str(JSON {found, paths:[{tables,joins,observed}]})` — internal-consistent (B1).
- `load_evalset() -> [{id,question,expect}]` consumed by `run_eval`; `judge_answer(question, answer, expect) -> {score,reason}` consumed by `run_eval` and matched by the test's `judge_fn` signature; `default_answer_fn(question) -> str` and `run_eval(..., answer_fn, judge_fn) -> {results, mean_score, pass_rate}` — consistent (C1↔C2↔C3).

**Cross-feature notes:** Feature C's `default_answer_fn` uses `answer_stream`, which has the P0 cache ON by default — for a clean eval, run with `query_cache_enabled=false` (set in `.env`) or accept cached repeats; the runner is deterministic given a fixed answer/judge. This is a usage note, not a code dependency.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-18-production-scaling-p1b.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
