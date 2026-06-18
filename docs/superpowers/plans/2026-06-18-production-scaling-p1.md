# Production-Scaling P1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the NeoCarta-Local semantic layer for sustained multi-user / large-corpus operation by adding Postgres connection pooling, bounded request concurrency, per-leg observability, incremental (content-hash) ingestion, and completing schema routing with fact-table selection.

**Architecture:** Four independent, additive features building on the merged P0 work. (A) A pooled Postgres connection (`agent/pg_pool.py`) replaces per-call `psycopg.connect`. (B) The controller gains a global concurrency gate + configurable leg parallelism + per-leg timing trace. (C) Ingestion gains a content-hash skip so unchanged documents are not re-parsed/re-embedded. (D) Schema routing (shipped OFF-by-default in P0) gains `select_fact_table` so the SQL leg's fact table comes from the routed set instead of the hardcoded sales fact.

**Tech Stack:** Python ≥3.11, Neo4j (`neo4j` driver), psycopg3 + psycopg_pool, LangChain (`openai:gpt-5.4-mini`), OpenAI embeddings, pydantic-settings, pytest.

## Global Constraints

- **Python:** `requires-python = ">=3.11"` (`backend/pyproject.toml`). New runtime deps must be added to `pyproject.toml` `[project].dependencies`.
- **Models/embeddings from config only:** chat via `semantic_layer.ingest.llm.get_chat_model(<settings id>)`, embeddings via `get_openai_client()` + `settings.embedding_model`/`embedding_dimensions`. Never hardcode model ids.
- **Tests run from `backend/` with the project venv:** `backend/.venv/bin/python -m pytest`. The Makefile's bare `python` is NOT on PATH — either activate the venv (`source backend/.venv/bin/activate`) or call `backend/.venv/bin/python` directly. `pythonpath = ["."]`.
- **Test markers (`backend/pyproject.toml`):** `neo4j`, `postgres`, `openai`. Marker-gated tests MUST take the matching fixture (`neo4j_driver`/`ingested_graph`, `postgres_dsn`, `require_openai`) so they SKIP when the service/key is absent. Pure-logic tests take no fixture and always run.
- **LLM stubbing convention (from `tests/test_legs.py`):** monkeypatch the module-level `get_chat_model` with a fake whose `.with_structured_output(...)` returns a `_FakeStructured` with a fixed `.invoke()` value. Never call a real model in a non-`openai` test.
- **Graph writes use `MERGE`** (mirroring `ingest/`).
- **⚠️ Running the full backend pytest suite wipes `Chunk.embedding`** (neo4j tests reset the shared graph). After a DB-touching run, restore with `source backend/.venv/bin/activate && make ingest` (or `backend/.venv/bin/python -m semantic_layer.ingest.pipeline`). Pure-logic tasks don't trigger this.
- **TDD + frequent commits:** every task is test-first and ends with a commit.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `backend/pyproject.toml` (modify) | add `psycopg-pool` dependency | A1 |
| `backend/semantic_layer/config.py` (modify) | new settings: pool sizes, concurrency, incremental flag | A1, B1, C2 |
| `backend/semantic_layer/agent/pg_pool.py` (create) | cached `ConnectionPool` for Postgres | A1 |
| `backend/semantic_layer/agent/sql_tools.py` (modify) | route `sales_pg` through the pool | A2 |
| `backend/semantic_layer/agent/controller.py` (modify) | concurrency gate, configurable leg workers, leg trace | B1, B2 |
| `backend/semantic_layer/ingest/doc_parser.py` (modify) | `file_content_hash`; include `file_hash` in parsed doc | C1 |
| `backend/semantic_layer/ingest/doc_loader.py` (modify) | persist `Document.file_hash`; `document_unchanged` | C1, C2 |
| `backend/semantic_layer/ingest/pipeline.py` (modify) | skip unchanged docs when `reset=False` | C2 |
| `backend/semantic_layer/agent/routing.py` (modify) | `select_fact_table(routed_tables)` | D1 |
| `backend/semantic_layer/agent/planner.py` (modify) | use selected fact table when routing enabled | D2 |
| `backend/tests/test_pg_pool.py` (create) | pool unit + postgres integration | A1, A2 |
| `backend/tests/test_controller_concurrency.py` (create) | gate + leg-worker config | B1 |
| `backend/tests/test_controller_trace.py` (create) | per-leg trace in answer event | B2 |
| `backend/tests/test_incremental_ingest.py` (create) | hash determinism + unchanged-skip | C1, C2 |
| `backend/tests/test_routing_fact_table.py` (create) | fact-table selection + planner wiring | D1, D2 |

Features A, C, D are mutually independent. Feature B's two tasks both modify `controller.py` and must be done in order (B1 then B2).

---

# Feature A — Postgres Connection Pooling

**Why:** `sql_tools._run` opens a fresh `psycopg.connect(settings.postgres_dsn)` on every SQL call (`sql_tools.py:24`). Under load that is a new TCP+auth handshake per query. A pooled connection amortizes this.

---

### Task A1: Cached connection pool + settings + dependency

**Files:**
- Modify: `backend/pyproject.toml`
- Modify: `backend/semantic_layer/config.py`
- Create: `backend/semantic_layer/agent/pg_pool.py`
- Test: `backend/tests/test_pg_pool.py`

**Interfaces:**
- Produces: `get_pool() -> psycopg_pool.ConnectionPool` (process-cached singleton, constructed with `open=False` so importing/constructing never connects). Reads `settings.pg_pool_min_size` / `settings.pg_pool_max_size`.

- [ ] **Step 1: Add the dependency**

In `backend/pyproject.toml`, in `[project].dependencies`, add after the `psycopg[binary]>=3.1` line:

```toml
    "psycopg-pool>=3.2",
```

Then install it into the venv:

Run: `cd backend && .venv/bin/python -m pip install "psycopg-pool>=3.2"`
Expected: installs `psycopg-pool` (and `psycopg-pool` only; psycopg is already present).

- [ ] **Step 2: Add settings**

In `backend/semantic_layer/config.py`, after the cache settings block (the `cache_similarity_threshold` line), add:

```python
    # Postgres connection pool (replaces per-call psycopg.connect in sql_tools).
    pg_pool_min_size: int = 1
    pg_pool_max_size: int = 8
```

- [ ] **Step 3: Write the failing test**

```python
# backend/tests/test_pg_pool.py
from semantic_layer.agent import pg_pool


def test_get_pool_is_cached_singleton():
    pg_pool.get_pool.cache_clear()
    p1 = pg_pool.get_pool()
    p2 = pg_pool.get_pool()
    assert p1 is p2                      # cached: one pool per process


def test_get_pool_uses_configured_sizes():
    from semantic_layer.config import settings
    pg_pool.get_pool.cache_clear()
    p = pg_pool.get_pool()
    assert p.max_size == settings.pg_pool_max_size
    assert p.min_size == settings.pg_pool_min_size
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_pg_pool.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'semantic_layer.agent.pg_pool'`.

- [ ] **Step 5: Implement**

```python
# backend/semantic_layer/agent/pg_pool.py
"""Process-wide pooled Postgres connections for the SQL tool.

Replaces sql_tools' per-call psycopg.connect with a ConnectionPool so repeated
queries reuse warm connections. Constructed with open=False so importing this
module (and unit tests) never touches the database; the pool opens lazily on
first use in sql_tools._run."""

from functools import lru_cache

from psycopg_pool import ConnectionPool

from semantic_layer.config import settings


@lru_cache
def get_pool() -> ConnectionPool:
    return ConnectionPool(
        conninfo=settings.postgres_dsn,
        min_size=settings.pg_pool_min_size,
        max_size=settings.pg_pool_max_size,
        open=False,
    )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_pg_pool.py -v`
Expected: PASS (2 passed). No DB needed — `open=False` means construction does not connect.

- [ ] **Step 7: Commit**

```bash
git add backend/pyproject.toml backend/semantic_layer/config.py backend/semantic_layer/agent/pg_pool.py backend/tests/test_pg_pool.py
git commit -m "feat(pg): cached Postgres connection pool"
```

---

### Task A2: Route `sales_pg` queries through the pool

**Files:**
- Modify: `backend/semantic_layer/agent/sql_tools.py:23-27`
- Test: `backend/tests/test_pg_pool.py`

**Interfaces:**
- Consumes: `get_pool()` (Task A1).
- Produces: `_run("sales_pg", sql, ...)` executes via a pooled connection; behavior (columns, rows, `agent_max_rows` limit, error JSON) is unchanged.

- [ ] **Step 1: Write the failing test (postgres-marked integration)**

```python
# append to backend/tests/test_pg_pool.py
import json

import pytest


@pytest.mark.postgres
def test_run_sales_pg_uses_pool_and_returns_rows(postgres_dsn, monkeypatch):
    from semantic_layer.agent import sql_tools
    from semantic_layer.agent import pg_pool

    pg_pool.get_pool.cache_clear()
    used = {"pool": False}
    real_get_pool = pg_pool.get_pool

    def tracking_get_pool():
        used["pool"] = True
        return real_get_pool()

    monkeypatch.setattr(sql_tools, "get_pool", tracking_get_pool)
    out = json.loads(sql_tools._run("sales_pg", "SELECT 1 AS n"))
    assert used["pool"] is True
    assert out["columns"] == ["n"]
    assert out["rows"] == [[1]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_pg_pool.py::test_run_sales_pg_uses_pool_and_returns_rows -v`
Expected: FAIL with `AttributeError: module 'semantic_layer.agent.sql_tools' has no attribute 'get_pool'` (or SKIP if Postgres is unavailable — if it skips, you cannot verify RED; proceed to implement and rely on the import-level check by running the unit tests in Task A1 plus a focused re-run when Postgres is available).

- [ ] **Step 3: Implement**

In `backend/semantic_layer/agent/sql_tools.py`, add the import near the top (after `import psycopg`):

```python
from semantic_layer.agent.pg_pool import get_pool
```

Replace the `sales_pg` branch (currently lines 23-27):

```python
        if source == "sales_pg":
            with psycopg.connect(settings.postgres_dsn) as conn, conn.cursor() as cur:
                cur.execute(sql) if params is None else cur.execute(sql, params)
                cols = [d.name for d in cur.description]
                rows = cur.fetchmany(limit)
```

with:

```python
        if source == "sales_pg":
            pool = get_pool()
            pool.open()  # idempotent; opens the pool lazily on first real use
            with pool.connection() as conn, conn.cursor() as cur:
                cur.execute(sql) if params is None else cur.execute(sql, params)
                cols = [d.name for d in cur.description]
                rows = cur.fetchmany(limit)
```

(The top-level `import psycopg` stays — it is still used for typing/errors and the module imports cleanly.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_pg_pool.py tests/test_agent_sql_tools.py -v`
Expected: pool unit tests PASS; the postgres-marked tests PASS (or SKIP if no Postgres). Existing `test_agent_sql_tools.py` stays green.

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/agent/sql_tools.py backend/tests/test_pg_pool.py
git commit -m "feat(pg): route sales_pg SQL through the connection pool"
```

---

# Feature B — Controller Hardening (concurrency gate + leg tracing)

**Why:** Every `/chat` request runs `answer_stream`, which spawns a `ThreadPoolExecutor(max_workers=4)` (`controller.py:62`) and makes synchronous LLM calls. With N concurrent requests this is unbounded fan-out into OpenAI; there is also no per-leg timing for diagnosing slow answers. Both tasks modify `controller.py`; do B1 then B2.

---

### Task B1: Global concurrency gate + configurable leg workers

**Files:**
- Modify: `backend/semantic_layer/config.py`
- Modify: `backend/semantic_layer/agent/controller.py`
- Test: `backend/tests/test_controller_concurrency.py`

**Interfaces:**
- Produces: module-level `_answer_gate` (a `threading.BoundedSemaphore(settings.max_concurrent_answers)`) held across the compute/stream section of `answer_stream`; a cache hit returns WITHOUT acquiring it. The leg `ThreadPoolExecutor` uses `settings.leg_max_workers`.

- [ ] **Step 1: Add settings**

In `backend/semantic_layer/config.py`, after the `pg_pool_max_size` line (Task A1) — or after the cache settings if Feature A is not done — add:

```python
    # Controller concurrency: max simultaneous answers computed server-wide, and
    # the per-answer leg thread-pool size.
    max_concurrent_answers: int = 8
    leg_max_workers: int = 4
```

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/test_controller_concurrency.py
from semantic_layer.agent import controller as ctrl
from semantic_layer.agent import cache as cache_mod
from semantic_layer.agent.planner import Intent


class _CountingGate:
    def __init__(self):
        self.acquired = 0
        self.released = 0

    def __enter__(self):
        self.acquired += 1
        return self

    def __exit__(self, *exc):
        self.released += 1
        return False


def _stub_live_run(monkeypatch):
    monkeypatch.setattr(ctrl, "extract_intent", lambda q: Intent(needs_sql=False))
    monkeypatch.setattr(ctrl, "build_plan", lambda intent, question=None: {
        "highlight": [], "sql_legs": [], "doc_leg": None, "api_correlations": []})
    monkeypatch.setattr(ctrl, "_synthesize", lambda *a, **k: "answer")
    monkeypatch.setattr(ctrl, "check_numeric_grounding", lambda *a, **k: [])


def test_live_run_acquires_and_releases_gate_once(monkeypatch):
    monkeypatch.setattr(ctrl, "query_cache", cache_mod.QueryCache(max_entries=10, ttl_seconds=1000))
    monkeypatch.setattr(ctrl.settings, "query_cache_enabled", False, raising=False)
    gate = _CountingGate()
    monkeypatch.setattr(ctrl, "_answer_gate", gate)
    _stub_live_run(monkeypatch)
    list(ctrl.answer_stream("anything"))
    assert gate.acquired == 1
    assert gate.released == 1


def test_cache_hit_does_not_acquire_gate(monkeypatch):
    fresh = cache_mod.QueryCache(max_entries=10, ttl_seconds=1000)
    monkeypatch.setattr(ctrl, "query_cache", fresh)
    monkeypatch.setattr(ctrl.settings, "query_cache_enabled", True, raising=False)
    gate = _CountingGate()
    monkeypatch.setattr(ctrl, "_answer_gate", gate)
    _stub_live_run(monkeypatch)
    list(ctrl.answer_stream("q"))        # populate (acquires once)
    assert gate.acquired == 1
    list(ctrl.answer_stream("q"))        # exact hit: must NOT acquire again
    assert gate.acquired == 1
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_controller_concurrency.py -v`
Expected: FAIL with `AttributeError: module 'semantic_layer.agent.controller' has no attribute '_answer_gate'`.

- [ ] **Step 4: Implement**

In `backend/semantic_layer/agent/controller.py`, add `import threading` to the imports at the top (next to `import json`). After the imports / before `_SYNTH_PROMPT`, add the module-level gate:

```python
_answer_gate = threading.BoundedSemaphore(settings.max_concurrent_answers)
```

In `answer_stream`, the cache-check block (which early-returns on a hit) stays exactly as-is BEFORE the gate. Wrap the compute-and-stream section in the gate. Concretely, change the structure so the `try:` ... through the final `yield answer_event` runs inside `with _answer_gate:`. The simplest correct edit: replace the line `    try:` (the one that begins the compute section, after the cache check and after the `collected`/`_emit` setup added in P0) with:

```python
    with _answer_gate:
      try:
```

and indent the entire body of that `try`/`except`/post-block (everything from `intent = extract_intent(...)` through the final `yield answer_event` / `else: yield answer_event`) by one extra level so it sits inside the `with`. The gate is released when the generator finishes or is closed (the `with` exits).

Also change the executor line (currently `with ThreadPoolExecutor(max_workers=4) as pool:`) to:

```python
        with ThreadPoolExecutor(max_workers=settings.leg_max_workers) as pool:
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_controller_concurrency.py tests/test_controller.py tests/test_query_cache.py -v`
Expected: new tests PASS; existing controller + cache tests stay green.

- [ ] **Step 6: Commit**

```bash
git add backend/semantic_layer/config.py backend/semantic_layer/agent/controller.py backend/tests/test_controller_concurrency.py
git commit -m "feat(controller): bounded concurrency gate + configurable leg workers"
```

---

### Task B2: Per-leg timing trace on the answer event

**Files:**
- Modify: `backend/semantic_layer/agent/controller.py`
- Test: `backend/tests/test_controller_trace.py`

**Interfaces:**
- Consumes: the leg fan-out loop in `answer_stream`.
- Produces: the success-path `answer_event` carries a `"trace"` key — a list of `{"name": str, "duration_ms": float, "ok": bool}`, one per executed leg, where `name` is the `kind`+`label` and `ok` is `error is None`. Existing answer fields are unchanged.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_controller_trace.py
from semantic_layer.agent import controller as ctrl
from semantic_layer.agent import cache as cache_mod
from semantic_layer.agent.planner import Intent


def test_answer_event_includes_per_leg_trace(monkeypatch):
    monkeypatch.setattr(ctrl, "query_cache", cache_mod.QueryCache(max_entries=10, ttl_seconds=1000))
    monkeypatch.setattr(ctrl.settings, "query_cache_enabled", False, raising=False)
    monkeypatch.setattr(ctrl, "extract_intent",
                        lambda q: Intent(needs_sql=True, needs_doc=True, doc_query="d"))
    monkeypatch.setattr(ctrl, "build_plan", lambda intent, question=None: {
        "highlight": [], "api_correlations": [],
        "sql_legs": [{"source": "sales_pg", "fact_table": "t", "join_targets": [],
                      "filters": [], "scope": {}}],
        "doc_leg": {"doc_query": "d", "candidate_doc_ids": [], "periods": []}})
    monkeypatch.setattr(ctrl, "run_sql_leg", lambda leg: {
        "source": "sales_pg", "sql": "SELECT 1", "columns": ["n"], "rows": [[1]],
        "row_count": 1, "error": None})
    monkeypatch.setattr(ctrl, "run_doc_leg", lambda q: {
        "answer": "a", "citations": [], "doc_texts": [], "error": None})
    monkeypatch.setattr(ctrl, "_synthesize", lambda *a, **k: "answer")
    monkeypatch.setattr(ctrl, "check_numeric_grounding", lambda *a, **k: [])

    answer = list(ctrl.answer_stream("q"))[-1]
    trace = answer["trace"]
    assert {t["name"] for t in trace} == {"sql:sales_pg", "doc:doc"}
    assert all(isinstance(t["duration_ms"], float) and t["duration_ms"] >= 0 for t in trace)
    assert all(t["ok"] is True for t in trace)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_controller_trace.py -v`
Expected: FAIL with `KeyError: 'trace'`.

- [ ] **Step 3: Implement**

In `backend/semantic_layer/agent/controller.py`, add `import time` to the top imports. In `answer_stream`, initialize a trace list alongside the result accumulators (where `sql_runs, api_calls, ... = [], [], ...` is set up):

```python
            trace = []
```

In the `for fut in list(jobs):` loop, wrap the `fut.result()` with timing. Replace:

```python
                kind, label = jobs[fut]
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001 — one leg failing must not sink the answer
```

with:

```python
                kind, label = jobs[fut]
                _t0 = time.perf_counter()
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001 — one leg failing must not sink the answer
```

and immediately after the `if kind == "sql": ... elif ... doc: ...` block that records `res` (i.e. after the result is folded in), append the trace entry:

```python
                trace.append({"name": f"{kind}:{label}",
                              "duration_ms": round((time.perf_counter() - _t0) * 1000, 2),
                              "ok": (res.get("error") is None)})
```

Add `"trace": trace` to the `answer_event` dict (alongside `caveats`):

```python
    answer_event = {"type": "answer", "content": summary, "highlight": plan.get("highlight", []),
                    "sql_runs": sql_runs, "api_calls": api_calls,
                    "doc_citations": doc_citations, "caveats": caveats, "trace": trace}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_controller_trace.py tests/test_controller.py tests/test_query_cache.py tests/test_web_chat_events.py -v`
Expected: new test PASS; existing controller/cache/web tests stay green (the extra `trace` key is additive and JSON-serializable, so the SSE layer passes it through).

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/agent/controller.py backend/tests/test_controller_trace.py
git commit -m "feat(controller): per-leg timing trace on answer event"
```

---

# Feature C — Incremental Ingestion (content-hash skip)

**Why:** `run_ingest` re-parses (liteparse + OCR) and re-embeds every PDF on every run (`pipeline.py:62-69`). For a large corpus this is the dominant cost. A content hash lets an incremental run skip documents whose source bytes are unchanged.

---

### Task C1: File content hash on the parsed document

**Files:**
- Modify: `backend/semantic_layer/ingest/doc_parser.py`
- Modify: `backend/semantic_layer/ingest/doc_loader.py`
- Test: `backend/tests/test_incremental_ingest.py`

**Interfaces:**
- Produces: `file_content_hash(path: str) -> str` (sha256 hex of the file bytes) in `doc_parser.py`; `parse_document` output gains a `"file_hash"` key; `load_document` persists `Document.file_hash`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_incremental_ingest.py
import hashlib

from semantic_layer.ingest.doc_parser import file_content_hash


def test_file_content_hash_is_stable_sha256(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert file_content_hash(str(p)) == expected
    assert file_content_hash(str(p)) == file_content_hash(str(p))  # deterministic


def test_file_content_hash_changes_with_content(tmp_path):
    a = tmp_path / "a.bin"; a.write_bytes(b"one")
    b = tmp_path / "b.bin"; b.write_bytes(b"two")
    assert file_content_hash(str(a)) != file_content_hash(str(b))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_incremental_ingest.py -v`
Expected: FAIL with `ImportError: cannot import name 'file_content_hash'`.

- [ ] **Step 3: Implement**

In `backend/semantic_layer/ingest/doc_parser.py`, add at the top (after `from pathlib import Path`):

```python
import hashlib
```

Add the helper above `parse_document`:

```python
def file_content_hash(path: str) -> str:
    """sha256 hex of a file's bytes — identity key for incremental ingestion."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
```

In `parse_document`, add `"file_hash"` to the returned dict:

```python
    return {
        "doc_id": doc_id,
        "title": Path(path).stem,
        "path": str(path),
        "num_pages": result.num_pages,
        "file_hash": file_content_hash(path),
        "chunks": chunks,
    }
```

In `backend/semantic_layer/ingest/doc_loader.py`, persist the hash. Change `_DOC_CYPHER`'s `SET` line for the document:

```python
SET d.title = $title, d.path = $path, d.num_pages = $num_pages, d.file_hash = $file_hash
```

and pass it in `load_document`:

```python
        session.run(
            _DOC_CYPHER,
            doc_id=doc["doc_id"], title=doc["title"], path=doc["path"],
            num_pages=doc["num_pages"], file_hash=doc.get("file_hash"), chunks=doc["chunks"],
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_incremental_ingest.py tests/test_doc_parser.py tests/test_doc_loader.py -v`
Expected: new hash tests PASS; existing parser/loader tests stay green (or SKIP for neo4j).

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/ingest/doc_parser.py backend/semantic_layer/ingest/doc_loader.py backend/tests/test_incremental_ingest.py
git commit -m "feat(ingest): file content hash on parsed/loaded documents"
```

---

### Task C2: Skip unchanged documents on incremental runs

**Files:**
- Modify: `backend/semantic_layer/config.py`
- Modify: `backend/semantic_layer/ingest/doc_loader.py`
- Modify: `backend/semantic_layer/ingest/pipeline.py`
- Test: `backend/tests/test_incremental_ingest.py`

**Interfaces:**
- Consumes: `file_content_hash` (C1), `settings.ingest_skip_unchanged`.
- Produces: `document_unchanged(driver, doc_id, file_hash) -> bool` in `doc_loader.py` (True iff a `Document` with that id has the same `file_hash` AND at least one of its chunks has an embedding). `run_ingest` skips parse+load for unchanged docs when `reset=False and settings.ingest_skip_unchanged`, counting them in `counts["documents_skipped"]`.

- [ ] **Step 1: Add setting**

In `backend/semantic_layer/config.py`, after the incremental-relevant settings (or after `ingest_max_workers`):

```python
    # Incremental ingestion: skip re-parsing/re-embedding documents whose source
    # bytes are unchanged (only consulted when run_ingest is called with reset=False).
    ingest_skip_unchanged: bool = True
```

- [ ] **Step 2: Write the failing test**

```python
# append to backend/tests/test_incremental_ingest.py
import pytest


@pytest.mark.neo4j
def test_document_unchanged_detects_matching_hash(ingested_graph):
    from semantic_layer.ingest.doc_loader import document_unchanged, load_document

    doc_id = "doc:incremental_probe"
    load_document(ingested_graph, {
        "doc_id": doc_id, "title": "probe", "path": "/tmp/probe.pdf",
        "num_pages": 1, "file_hash": "abc123",
        "chunks": [{"chunk_id": f"{doc_id}:chunk:0", "doc_id": doc_id, "ordinal": 0, "text": "x"}],
    })
    # No embedding on the chunk yet -> not "unchanged" (would need re-embed).
    assert document_unchanged(ingested_graph, doc_id, "abc123") is False
    # Give the chunk an embedding, then a matching hash is "unchanged".
    with ingested_graph.session() as s:
        s.run("MATCH (c:Chunk {id:$id}) CALL db.create.setNodeVectorProperty(c,'embedding',$v)",
              id=f"{doc_id}:chunk:0", v=[0.1, 0.2, 0.3])
    assert document_unchanged(ingested_graph, doc_id, "abc123") is True
    assert document_unchanged(ingested_graph, doc_id, "different") is False
    # cleanup
    with ingested_graph.session() as s:
        s.run("MATCH (d:Document {id:$id}) OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c) DETACH DELETE d, c",
              id=doc_id)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_incremental_ingest.py::test_document_unchanged_detects_matching_hash -v`
Expected: FAIL with `ImportError: cannot import name 'document_unchanged'` (or SKIP if Neo4j unavailable).

- [ ] **Step 4: Implement**

In `backend/semantic_layer/ingest/doc_loader.py`, add:

```python
_UNCHANGED_CYPHER = """
MATCH (d:Document {id: $doc_id})
WHERE d.file_hash = $file_hash
MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
WHERE c.embedding IS NOT NULL
RETURN count(c) AS embedded
"""


def document_unchanged(driver: Driver, doc_id: str, file_hash: str) -> bool:
    """True iff this document exists with the same file_hash and has embedded chunks
    (so skipping re-ingest is safe — nothing to re-parse or re-embed)."""
    with driver.session(database=settings.neo4j_database) as session:
        rec = session.run(_UNCHANGED_CYPHER, doc_id=doc_id, file_hash=file_hash).single()
    return bool(rec and rec["embedded"] > 0)
```

In `backend/semantic_layer/ingest/pipeline.py`, add `from semantic_layer.ingest.doc_parser import parse_document, file_content_hash` (extend the existing `parse_document` import) and `from semantic_layer.ingest.doc_loader import load_document, document_unchanged` (extend the existing import). Replace the document loop (currently lines 62-69):

```python
        docs_dir = Path(settings.docs_dir)
        pdfs = sorted(docs_dir.glob("*.pdf"))
        for pdf in pdfs:
            doc = parse_document(str(pdf))
            load_document(driver, doc)
            # Deterministic period extraction (regex) — runs without the LLM.
            link_document_period(driver, doc["doc_id"], extract_period(doc))
        counts["documents"] = len(pdfs)
```

with:

```python
        docs_dir = Path(settings.docs_dir)
        pdfs = sorted(docs_dir.glob("*.pdf"))
        ingested, skipped = 0, 0
        for pdf in pdfs:
            doc_id = f"doc:{pdf.stem}"
            if not reset and settings.ingest_skip_unchanged and \
                    document_unchanged(driver, doc_id, file_content_hash(str(pdf))):
                skipped += 1
                continue
            doc = parse_document(str(pdf))
            load_document(driver, doc)
            # Deterministic period extraction (regex) — runs without the LLM.
            link_document_period(driver, doc["doc_id"], extract_period(doc))
            ingested += 1
        counts["documents"] = ingested
        counts["documents_skipped"] = skipped
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_incremental_ingest.py tests/test_pipeline.py -v`
Expected: new neo4j test PASS (or SKIP); `test_pipeline.py` stays green (it calls `run_ingest` with `reset=True` by default, so the skip path is inert there). If you ran a DB-touching suite, restore embeddings afterward (`make ingest`).

- [ ] **Step 6: Commit**

```bash
git add backend/semantic_layer/config.py backend/semantic_layer/ingest/doc_loader.py backend/semantic_layer/ingest/pipeline.py backend/tests/test_incremental_ingest.py
git commit -m "feat(ingest): skip unchanged documents on incremental runs"
```

---

# Feature D — Schema-Routing Completion (fact-table selection)

**Why:** P0 shipped `route_tables` but `build_plan` still hardcodes the SQL fact table to `_SALES_FACT = "table:sales_pg.sales.order_line"` (`planner.py:58`). To make routing actually choose where the measure comes from, pick the fact table from the routed set (the routed sales-schema table that is the hub of the star — the one with the most foreign keys).

---

### Task D1: `select_fact_table` from the routed set

**Files:**
- Modify: `backend/semantic_layer/agent/routing.py`
- Test: `backend/tests/test_routing_fact_table.py`

**Interfaces:**
- Produces: `select_fact_table(routed_tables: list[str]) -> str | None` — among `routed_tables`, returns the sales-schema table with the most outgoing `REFERENCES` (FK) edges (the fact at the centre of the star schema); `None` if `routed_tables` is empty or none are sales-schema tables.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_routing_fact_table.py
import pytest

from semantic_layer.agent import routing


def test_select_fact_table_empty_returns_none():
    assert routing.select_fact_table([]) is None


@pytest.mark.neo4j
def test_select_fact_table_picks_order_line(ingested_graph):
    # order_line is the fact (most FKs); region is a leaf dimension (no FKs out).
    routed = ["table:sales_pg.sales.region", "table:sales_pg.sales.order_line",
              "table:sales_pg.sales.customer"]
    assert routing.select_fact_table(routed) == "table:sales_pg.sales.order_line"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/python -m pytest tests/test_routing_fact_table.py -v`
Expected: FAIL with `AttributeError: module 'semantic_layer.agent.routing' has no attribute 'select_fact_table'`.

- [ ] **Step 3: Implement**

In `backend/semantic_layer/agent/routing.py`, add the import (near the existing imports):

```python
from semantic_layer.agent.driver import driver
```

Add:

```python
_FACT_RANK_CYPHER = """
UNWIND $tables AS tid
MATCH (t:Table {id: tid})
WHERE tid STARTS WITH 'table:sales_pg.sales.'
OPTIONAL MATCH (t)-[:HAS_COLUMN]->(:Column)-[:REFERENCES]->(:Column)
WITH tid, count(*) AS fks
RETURN tid ORDER BY fks DESC, tid LIMIT 1
"""


def select_fact_table(routed_tables: list[str]) -> str | None:
    """Pick the SQL fact table from a routed set: the sales-schema table with the
    most foreign keys (the hub of the star). Returns None when none qualify."""
    sales = [t for t in routed_tables if t.startswith("table:sales_pg.sales.")]
    if not sales:
        return None
    from semantic_layer.config import settings
    recs = driver().execute_query(
        _FACT_RANK_CYPHER, tables=sales, database_=settings.neo4j_database,
    ).records
    return recs[0]["tid"] if recs else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_routing_fact_table.py -v`
Expected: empty-case PASS; neo4j case PASS (or SKIP if Neo4j unavailable).

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/agent/routing.py backend/tests/test_routing_fact_table.py
git commit -m "feat(routing): select_fact_table from the routed set"
```

---

### Task D2: Use the selected fact table in `build_plan`

**Files:**
- Modify: `backend/semantic_layer/agent/planner.py`
- Test: `backend/tests/test_routing_fact_table.py`

**Interfaces:**
- Consumes: `select_fact_table` (D1), the existing `routed_tables` computed in `build_plan` (P0).
- Produces: when `settings.schema_routing_enabled` and `select_fact_table(routed_tables)` returns a table, the sales `sql_leg`'s `fact_table` is that table (and its `join_targets`/`columns` are computed from it); otherwise `_SALES_FACT` is used (unchanged default-off behavior).

- [ ] **Step 1: Write the failing test**

```python
# append to backend/tests/test_routing_fact_table.py
from semantic_layer.agent import planner as planner_mod
from semantic_layer.agent.planner import Intent, build_plan


def _stub_graph(monkeypatch, dim_targets):
    monkeypatch.setattr(planner_mod, "_resolve_values", lambda terms: [])
    monkeypatch.setattr(planner_mod, "_dimension_targets", lambda gb: dim_targets)
    monkeypatch.setattr(planner_mod, "_join_targets",
                        lambda fact, ids: [{"table_id": t, "tables": [fact, t], "joins": []}
                                           for t in dict.fromkeys(ids)])
    monkeypatch.setattr(planner_mod, "_table_columns", lambda tid: ["amount"])
    monkeypatch.setattr(planner_mod, "_context_docs", lambda terms: None)
    monkeypatch.setattr(planner_mod, "_api_correlations", lambda: [])


def test_build_plan_uses_selected_fact_table_when_routing_on(monkeypatch):
    _stub_graph(monkeypatch, dim_targets=["table:sales_pg.sales.segment"])
    monkeypatch.setattr(planner_mod.settings, "schema_routing_enabled", True, raising=False)
    monkeypatch.setattr(planner_mod, "route_tables",
                        lambda q, k_ret, k_rank: ["table:sales_pg.sales.invoice_line"])
    monkeypatch.setattr(planner_mod, "select_fact_table",
                        lambda routed: "table:sales_pg.sales.invoice_line")
    plan = build_plan(Intent(group_by=["segment"], needs_sql=True), question="revenue by segment")
    assert plan["sql_legs"][0]["fact_table"] == "table:sales_pg.sales.invoice_line"


def test_build_plan_keeps_default_fact_when_routing_off(monkeypatch):
    _stub_graph(monkeypatch, dim_targets=["table:sales_pg.sales.segment"])
    monkeypatch.setattr(planner_mod.settings, "schema_routing_enabled", False, raising=False)
    plan = build_plan(Intent(group_by=["segment"], needs_sql=True))
    assert plan["sql_legs"][0]["fact_table"] == planner_mod._SALES_FACT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_routing_fact_table.py::test_build_plan_uses_selected_fact_table_when_routing_on -v`
Expected: FAIL — `fact_table` is `_SALES_FACT`, not the selected table (and `AttributeError` for `planner_mod.select_fact_table` until imported).

- [ ] **Step 3: Implement**

In `backend/semantic_layer/agent/planner.py`, extend the routing import:

```python
from semantic_layer.agent.routing import route_tables, select_fact_table
```

In `build_plan`, after `routed_tables` is computed (the `if settings.schema_routing_enabled and question:` block), determine the fact table:

```python
    fact_table = _SALES_FACT
    if routed_tables:
        fact_table = select_fact_table(routed_tables) or _SALES_FACT
```

Then in the sales `sql_legs.append({...})` block, replace the two `_SALES_FACT` references (`"fact_table": _SALES_FACT` and `"join_targets": _join_targets(_SALES_FACT, sales_target_ids)`) with `fact_table`:

```python
        sql_legs.append({
            "source": "sales_pg",
            "fact_table": fact_table,
            "join_targets": _join_targets(fact_table, sales_target_ids),
            "filters": [{"table_id": r["table_id"], "column": r["column"], "value": r["exact"]}
                        for r in sales_dims],
            "group_by": list(intent.group_by) if dim_targets else [],
            "scope": scope,
        })
```

(When routing is off, `routed_tables` is `[]`, so `fact_table` stays `_SALES_FACT` — behavior unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_routing_fact_table.py tests/test_planner_plan.py tests/test_planner_routing.py -v`
Expected: new tests PASS; existing planner tests stay green (the one known neo4j doc-embedding failure aside, if the suite wiped embeddings).

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/agent/planner.py backend/tests/test_routing_fact_table.py
git commit -m "feat(routing): build_plan uses selected fact table when routing enabled"
```

---

## Deferred to a separate plan (P1b) — and why

These two P1 items are real and valuable but are each their own subsystem that needs live iteration a no-placeholder plan should not guess at. Track them as a follow-up plan, not tasks here:

1. **Weighted k-shortest-path join discovery** (replace `allShortestPaths` in `graph_tools.get_join_path:118-140`). Getting correct, non-explosive Cypher for *k* paths beyond the minimal length (Yen's / `apoc.path` expansion with a hop+weight cost) requires iterating against the live Neo4j with APOC and profiling on the real schema. Spec it after measuring where the current 24-hop shortest-path actually fails.
2. **Offline LLM-judge eval harness** (a runnable scorer over an expanded golden-question set with a 1–4 rubric, à la LinkedIn). This needs a judge-model decision, a golden-set expansion, and a reporting format — a substantial subsystem. It also unlocks *tuning* schema routing (`k_ret`/`k_rank`/`min_score`) and measuring cache hit-rate, so it pairs naturally with a "routing tuning" follow-up.

---

## Self-Review

**Spec coverage** (against the P1 list from the scalability assessment):
- ✅ Postgres connection pooling — Feature A.
- ✅ Async/queue concurrency — Feature B1 (bounded gate + configurable leg workers; a pragmatic threaded increment, not a full async rewrite).
- ✅ Observability — Feature B2 (per-leg trace; structured timing on the answer event).
- ✅ Incremental/CDC ingestion — Feature C.
- ✅ Schema-routing completion (recommended next step) — Feature D.
- ⏸️ Weighted k-shortest-path joins, full eval harness — explicitly deferred with rationale above.

**Placeholder scan:** No TBDs; every code step has complete code; every run step states the expected result and the venv command.

**Type consistency:**
- `get_pool() -> ConnectionPool` consumed by `sql_tools._run` via `get_pool().connection()` — consistent (A1↔A2).
- `_answer_gate` (context-manager semaphore) and `settings.leg_max_workers` used in `answer_stream` — consistent (B1); `trace` list of `{name,duration_ms,ok}` added to `answer_event` — consistent (B2).
- `file_content_hash(path) -> str` consumed by `parse_document` and `pipeline`; `document_unchanged(driver, doc_id, file_hash) -> bool` consumed by `pipeline` — consistent (C1↔C2).
- `select_fact_table(routed_tables) -> str | None` consumed by `build_plan` — consistent (D1↔D2).

**Cross-feature notes:** Features B2 and the P0 cache compose — the cache stores the full event list including the `trace`-bearing answer event, so cached replays carry the trace. Feature B1's gate sits *after* the cache check, so cache hits never consume a concurrency slot.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-18-production-scaling-p1.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
