# Cross-Source Bridge Edges Implementation Plan (Plan 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Encode API↔SQL entity links in the graph (`account_id —SAME_ENTITY→ customer_id`) so join-path planning can fold API endpoints into cross-source queries.

**Architecture:** A deterministic ingest step adds `SAME_ENTITY` edges from each REST-API column named `account_id` to `sales.customer.customer_id`. `get_join_path` is taught to traverse that relationship, so the existing agent immediately gains API↔SQL join awareness. No LLM involved.

**Tech Stack:** Python 3.14 + pytest (run as `cd backend && .venv/bin/python -m pytest …`), Neo4j (Cypher via the `neo4j` driver). Live-graph tests are `@pytest.mark.neo4j` and use the `ingested_graph` fixture (Docker services must be up; the fixture runs `run_ingest`).

**Spec:** [Graph-Native Planner + Parallel Legs](../specs/2026-06-17-graph-native-planner-design.md) — "Cross-source bridge edges" section.

---

## File Structure

- **Create** `backend/semantic_layer/ingest/bridge.py` — `bridge_sources(driver) -> int`, the deterministic bridge-edge step.
- **Create** `backend/tests/test_bridge.py` — verifies the edges exist after ingest.
- **Modify** `backend/semantic_layer/ingest/pipeline.py` — call `bridge_sources` in `run_ingest` after `index_periods`.
- **Modify** `backend/semantic_layer/agent/graph_tools.py:get_join_path` — traverse `SAME_ENTITY` too.
- **Modify** `backend/tests/test_agent_join_path.py` — add a bridge-traversal test.

---

## Task 1: `bridge_sources` ingest step

**Files:**
- Create: `backend/semantic_layer/ingest/bridge.py`
- Create: `backend/tests/test_bridge.py`
- Modify: `backend/semantic_layer/ingest/pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_bridge.py
import pytest

from semantic_layer.config import settings


@pytest.mark.neo4j
def test_account_id_bridges_to_customer_id(ingested_graph):
    # ingested_graph runs run_ingest, which now calls bridge_sources.
    from semantic_layer.agent.driver import driver
    rows = driver().execute_query(
        """
        MATCH (ac:Column)-[:SAME_ENTITY]->(c:Column {id:'col:sales_pg.sales.customer.customer_id'})
        RETURN ac.id AS api_col ORDER BY api_col
        """,
        database_=settings.neo4j_database,
    ).records
    api_cols = [r["api_col"] for r in rows]
    # Every REST-API endpoint that carries account_id should now bridge to the customer PK.
    assert "col:itsm.api.GET /tickets.account_id" in api_cols
    assert "col:dgx.api.GET /usage.account_id" in api_cols
    assert all(c.endswith(".account_id") for c in api_cols)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_bridge.py -v`
Expected: FAIL — no `SAME_ENTITY` edges exist yet (empty `api_cols`), so the first assert fails. (If Docker/Neo4j is down the test skips; bring services up first.)

- [ ] **Step 3: Write the implementation**

Create `backend/semantic_layer/ingest/bridge.py`:

```python
"""Add cross-source bridge edges so API tables join to SQL tables in the graph.

REST APIs expose keys (e.g. account_id) that denote the same entity as a SQL primary
key (sales.customer.customer_id) but carry no modeled foreign key. We add explicit
SAME_ENTITY edges from each API key column to its SQL counterpart so join-path planning
can fold API endpoints into cross-source queries. Deterministic; no LLM, no inference.
"""

from semantic_layer.config import settings

# Explicit key map: an API column with this name denotes the given SQL column id.
_BRIDGES = {
    "account_id": "col:sales_pg.sales.customer.customer_id",
}

_CYPHER = """
UNWIND $pairs AS pair
MATCH (db:Database)-[:HAS_SCHEMA]->(:Schema)-[:HAS_TABLE]->(:Table)-[:HAS_COLUMN]->(ac:Column)
  WHERE db.platform = 'REST-API' AND ac.name = pair.key
MATCH (sql:Column {id: pair.target})
MERGE (ac)-[:SAME_ENTITY]->(sql)
RETURN count(*) AS n
"""


def bridge_sources(driver) -> int:
    """MERGE SAME_ENTITY edges from API key columns to their SQL counterparts.

    Returns the number of (API column, SQL column) pairs linked. Idempotent."""
    pairs = [{"key": key, "target": target} for key, target in _BRIDGES.items()]
    records = driver.execute_query(
        _CYPHER, pairs=pairs, database_=settings.neo4j_database,
    ).records
    return records[0]["n"] if records else 0
```

Then wire it into `backend/semantic_layer/ingest/pipeline.py`. Add the import after the
`period_indexer` import (line 18):

```python
from semantic_layer.ingest.bridge import bridge_sources
```

And add the call in `run_ingest` immediately after the `index_periods` line (`counts["periods"] = index_periods(driver)`, line 52):

```python
        # Link API key columns (account_id) to their SQL counterpart (customer_id) so
        # join-path planning can fold REST endpoints into cross-source queries.
        counts["bridges"] = bridge_sources(driver)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_bridge.py -v`
Expected: PASS (the `ingested_graph` fixture re-runs `run_ingest`, which now creates the edges).

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/ingest/bridge.py backend/tests/test_bridge.py backend/semantic_layer/ingest/pipeline.py
git commit -m "feat(ingest): SAME_ENTITY bridge edges from API account_id to customer_id"
```

---

## Task 2: Make `get_join_path` bridge-aware

**Files:**
- Modify: `backend/semantic_layer/agent/graph_tools.py:get_join_path`
- Test: `backend/tests/test_agent_join_path.py`

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_agent_join_path.py` (it already imports `json`, `pytest`, and `get_join_path`):

```python
@pytest.mark.neo4j
def test_join_path_bridges_sql_to_api(ingested_graph):
    # The SAME_ENTITY bridge lets order_line reach an API endpoint via customer.
    result = json.loads(get_join_path.invoke({
        "table_a_id": "table:sales_pg.sales.order_line",
        "table_b_id": "table:itsm.api.GET /tickets",
    }))
    assert result["found"] is True
    assert "table:sales_pg.sales.customer" in result["tables"]
    assert result["tables"][-1] == "table:itsm.api.GET /tickets"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_agent_join_path.py::test_join_path_bridges_sql_to_api -v`
Expected: FAIL with `result["found"] is False` — `get_join_path` traverses only `HAS_COLUMN|REFERENCES`, so it can't cross the `SAME_ENTITY` bridge.

- [ ] **Step 3: Add `SAME_ENTITY` to the traversal**

In `backend/semantic_layer/agent/graph_tools.py`, in `get_join_path`, change the
`shortestPath` relationship pattern from:

```cypher
        MATCH p = shortestPath((ta)-[:HAS_COLUMN|REFERENCES*1..24]-(tb))
```

to:

```cypher
        MATCH p = shortestPath((ta)-[:HAS_COLUMN|REFERENCES|SAME_ENTITY*1..24]-(tb))
```

Also update the tool docstring's first line to reflect bridges — change:

```python
    """Find the shortest foreign-key join path between two tables (by id).
```

to:

```python
    """Find the shortest join path between two tables (by id), across FK and cross-source bridges.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_agent_join_path.py -v`
Expected: PASS — all four tests (the original deep-join, disconnected, same-table guard, and the new bridge test).

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/agent/graph_tools.py backend/tests/test_agent_join_path.py
git commit -m "feat(agent): get_join_path traverses SAME_ENTITY bridges (API<->SQL)"
```

---

## Self-Review notes

- **Spec coverage:** "Cross-source bridge edges" section → Task 1 (`bridge_sources` + wiring) and its test ✓; spec's "`get_join_path(order_line, itsm GET /tickets)` returns a path through the bridge" → Task 2 ✓.
- **Idempotency:** `MERGE` makes `bridge_sources` safe on re-ingest (and the graph is reset each ingest anyway).
- **Platform value:** uses the verified stored value `'REST-API'` (uppercase) for the API-database filter.
- **Scope:** only the `account_id ↔ customer_id` bridge (per spec; key-map is extensible). No value-overlap inference.
- **Dependency:** `bridge_sources` runs after `load_bundle` (API columns loaded) and after `index_periods`; needs no LLM, so it runs in both `with_llm` modes.
- **Note for executor:** running backend pytest can wipe doc-chunk embeddings; after this plan run `make ingest` before any doc-dependent work. These two tasks don't touch embeddings.
```
