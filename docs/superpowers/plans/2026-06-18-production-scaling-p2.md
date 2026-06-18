# Production-Scaling P2 (Scale-Out) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the semantic layer scale across all three modalities at once — bound the whole-graph reads that blow up at thousands of documents, federate the agent over many enterprise APIs instead of four hardcoded ones, and activate + measure schema routing so thousands of tables across many databases stay tractable.

**Architecture:** Three independent application-layer features. (A) `get_schema_graph` gains a `source` filter + a hard chunk cap with a `truncated` flag, and `/graph` exposes them — so the UI requests bounded subgraphs, not the entire graph. (B) API registration and the `call_api` allow-list become config-driven (`settings.api_source_list`), and the API leg is fed graph-routed endpoints (`route_api_endpoints`) instead of a static prompt — the same retrieve-then-use pattern schema routing uses for tables. (C) When schema routing is enabled, `build_plan` bounds its join targets to the routed top-K, and a new A/B eval comparison (`eval/compare.py`) measures routing-off vs routing-on on the golden set so the default-on flip is a data-backed decision.

**Tech Stack:** Python ≥3.11, Neo4j, FastAPI, LangChain (`openai:gpt-5.4-mini`), pydantic-settings, pytest.

## Global Constraints

- **Python:** `requires-python = ">=3.11"` (`backend/pyproject.toml`). No new runtime dependency without adding it to `pyproject.toml`.
- **Models from config only:** chat via `semantic_layer.ingest.llm.get_chat_model(<settings id>)`. Never hardcode model ids.
- **Tests run from `backend/` with the project venv:** `backend/.venv/bin/python -m pytest`. The Makefile's bare `python` is NOT on PATH. `pythonpath = ["."]`.
- **Test markers (`backend/pyproject.toml`):** `neo4j`, `postgres`, `openai`. Marker-gated tests MUST take the matching fixture (`neo4j_driver`/`ingested_graph`, `postgres_dsn`, `require_openai`) so they SKIP when the service/key is absent. Pure-logic tests take no fixture.
- **LLM stubbing convention (from `tests/test_legs.py`):** monkeypatch the module-level `get_chat_model` with a fake whose `.with_structured_output(...)` returns a `_FakeStructured` with a fixed `.invoke()` value. Never call a real model in a non-`openai` test.
- **Graph writes use `MERGE`**; the new Cypher in this plan is read-only.
- **Backward compatibility:** every change must preserve current behavior at default settings (schema routing stays default-OFF; the 4 demo APIs stay the default registry; `/graph` with no params still renders the demo graph).
- **⚠️ Running the full backend pytest suite wipes `Chunk.embedding`.** After a DB-touching run, restore with `source backend/.venv/bin/activate && make ingest` (or `backend/.venv/bin/python -m semantic_layer.ingest.pipeline`).
- **TDD + frequent commits:** every task is test-first and ends with a commit.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `backend/semantic_layer/config.py` (modify) | `graph_max_chunks`, `api_sources`+`api_source_list`, `schema_routing_max_targets` | A1, B1, C1 |
| `backend/semantic_layer/web/graph_api.py` (modify) | bounded `get_schema_graph(source, max_chunks)` + `truncated` | A1 |
| `backend/semantic_layer/web/app.py` (modify) | `/graph` query params | A2 |
| `backend/semantic_layer/agent/api_tools.py` (modify) | config-driven source allow-list | B1 |
| `backend/semantic_layer/ingest/pipeline.py` (modify) | config-driven API registration | B1 |
| `backend/semantic_layer/agent/routing.py` (modify) | `route_api_endpoints(intents)` | B2 |
| `backend/semantic_layer/agent/legs.py` (modify) | API leg uses routed endpoints | B2 |
| `backend/semantic_layer/agent/planner.py` (modify) | bound routed join targets when routing on | C1 |
| `backend/semantic_layer/eval/compare.py` (create) | routing A/B comparison + CLI | C2 |
| `backend/tests/test_graph_api_bounded.py` (create) | bounded graph + endpoint params | A1, A2 |
| `backend/tests/test_api_federation.py` (create) | config registry + endpoint routing | B1, B2 |
| `backend/tests/test_routing_activation.py` (create) | bounded targets + A/B compare | C1, C2 |

Features A, B, C are mutually independent. Within each, do the tasks in order.

---

# Feature A — Bounded Graph Reads

**Why:** `get_schema_graph` (`graph_api.py:22`) returns the ENTIRE graph — every source, table, document, **every chunk**, every entity and bridge edge — and `/graph` is polled by the UI (visible in the runtime logs). At thousands of documents the chunk layer alone is tens of thousands of nodes per response. Bound it and let the UI request a per-source subgraph.

---

### Task A1: Cap the chunk layer + `source` filter + `truncated` flag

**Files:**
- Modify: `backend/semantic_layer/config.py`
- Modify: `backend/semantic_layer/web/graph_api.py`
- Test: `backend/tests/test_graph_api_bounded.py`

**Interfaces:**
- Produces: `get_schema_graph(source: str | None = None, max_chunks: int | None = None) -> dict` — returns `{nodes, edges, truncated: bool}`. `source` (a Database name) restricts the structured layer to that source's tables; documents/chunks are included only when `source` is None or `source == "documents"`. The chunk layer is capped at `max_chunks` (default `settings.graph_max_chunks`); entity/bridge edges are computed only over the included chunks. `truncated` is True when the cap dropped chunks. `get_sources()` is unchanged.

- [ ] **Step 1: Add the setting**

In `backend/semantic_layer/config.py`, after the existing graph/agent settings (e.g. after `agent_recursion_limit`), add:

```python
    # Bounded /graph rendering: cap the document chunk layer so the UI payload stays
    # renderable at thousands of documents (the structured layer is returned in full).
    graph_max_chunks: int = 300
```

- [ ] **Step 2: Write the failing tests**

```python
# backend/tests/test_graph_api_bounded.py
import pytest

from semantic_layer.web import graph_api


@pytest.mark.neo4j
def test_get_schema_graph_caps_chunks_and_sets_truncated(ingested_graph, monkeypatch):
    full = graph_api.get_schema_graph()
    assert "truncated" in full
    chunk_nodes = [n for n in full["nodes"] if n["kind"] == "chunk"]
    # With a tiny cap, the chunk layer is limited and truncated flips True.
    capped = graph_api.get_schema_graph(max_chunks=1)
    capped_chunks = [n for n in capped["nodes"] if n["kind"] == "chunk"]
    assert len(capped_chunks) <= 1
    if len(chunk_nodes) > 1:
        assert capped["truncated"] is True
    # Every edge references a node that is present (no dangling chunk/entity edges).
    ids = {n["id"] for n in capped["nodes"]}
    for e in capped["edges"]:
        assert e["source"] in ids and e["target"] in ids


@pytest.mark.neo4j
def test_get_schema_graph_source_filter_excludes_other_sources(ingested_graph):
    out = graph_api.get_schema_graph(source="sales_pg")
    sources = {n.get("source") for n in out["nodes"] if n["kind"] == "table"}
    assert sources <= {"sales_pg"}            # only sales_pg tables
    assert not [n for n in out["nodes"] if n["kind"] == "chunk"]  # docs excluded under a SQL source
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_graph_api_bounded.py -v`
Expected: FAIL — `get_schema_graph()` has no `truncated` key / `get_schema_graph(max_chunks=...)` is an unexpected kwarg (or SKIP if Neo4j unavailable).

- [ ] **Step 4: Implement**

In `backend/semantic_layer/web/graph_api.py`, add the import for settings is already present. Replace the `get_schema_graph` function body with the bounded version:

```python
def get_schema_graph(source: str | None = None, max_chunks: int | None = None) -> dict:
    """Renderable graph for the UI, BOUNDED for scale.

    `source` (a Database name) restricts the structured layer to that source's tables;
    the document layer is included only when source is None or 'documents'. The chunk
    layer is capped at max_chunks (default settings.graph_max_chunks); entity/bridge
    edges are computed only over the included chunks. Returns {nodes, edges, truncated}."""
    cap = max_chunks if max_chunks is not None else settings.graph_max_chunks
    include_docs = source is None or source == "documents"
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    truncated = False

    db_filter = "WHERE d.name = $source" if (source and source != "documents") else ""
    db_rows = driver().execute_query(
        f"MATCH (d:Database) {db_filter} RETURN d.id AS id, d.name AS name, d.platform AS platform",
        source=source, database_=settings.neo4j_database,
    ).records
    for r in db_rows:
        platform = (r["platform"] or "").upper()
        nodes[r["id"]] = {"id": r["id"], "label": r["name"], "kind": "source",
                          "source": r["name"],
                          "platform": "sql" if platform in _SQL_PLATFORMS else "api"}

    tbl_filter = "WHERE d.name = $source" if (source and source != "documents") else ""
    tbl_rows = driver().execute_query(
        f"""
        MATCH (d:Database)-[:HAS_SCHEMA]->(:Schema)-[:HAS_TABLE]->(t:Table)
        {tbl_filter}
        RETURN t.id AS id, t.name AS name, d.id AS db_id, d.name AS source
        """,
        source=source, database_=settings.neo4j_database,
    ).records
    for r in tbl_rows:
        nodes[r["id"]] = {"id": r["id"], "label": r["name"], "kind": "table", "source": r["source"]}
        edges.append({"source": r["db_id"], "target": r["id"], "type": "HAS_TABLE"})

    ref_rows = driver().execute_query(
        """
        MATCH (t1:Table)-[:HAS_COLUMN]->(:Column)-[:REFERENCES]->(:Column)<-[:HAS_COLUMN]-(t2:Table)
        WHERE t1 <> t2
        RETURN DISTINCT t1.id AS a, t2.id AS b
        """,
        database_=settings.neo4j_database,
    ).records
    for r in ref_rows:
        if r["a"] in nodes and r["b"] in nodes:   # only edges between included tables
            edges.append({"source": r["a"], "target": r["b"], "type": "REFERENCES"})

    if not include_docs:
        return {"nodes": list(nodes.values()), "edges": edges, "truncated": truncated}

    doc_rows = driver().execute_query(
        "MATCH (d:Document) RETURN d.id AS id, d.title AS title",
        database_=settings.neo4j_database,
    ).records
    for r in doc_rows:
        nodes[r["id"]] = {"id": r["id"], "label": r["title"], "kind": "document", "source": "documents"}

    chunk_rows = driver().execute_query(
        """
        MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
        RETURN c.id AS id, c.ordinal AS ordinal, c.text AS text, d.id AS doc_id
        ORDER BY d.id, c.ordinal
        LIMIT $cap_plus
        """,
        cap_plus=cap + 1, database_=settings.neo4j_database,
    ).records
    if len(chunk_rows) > cap:
        truncated = True
        chunk_rows = chunk_rows[:cap]
    chunk_ids = [r["id"] for r in chunk_rows]
    for r in chunk_rows:
        nodes[r["id"]] = {"id": r["id"], "label": f"¶{r['ordinal']}", "kind": "chunk",
                          "source": "documents", "text": (r["text"] or "")[:280]}
        edges.append({"source": r["doc_id"], "target": r["id"], "type": "HAS_CHUNK"})

    ent_rows = driver().execute_query(
        """
        MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
        WHERE c.id IN $chunk_ids
        WITH e, collect(DISTINCT c.id) AS chunk_ids
        WHERE size(chunk_ids) >= 2 OR exists((e)-[:REFERS_TO]->(:Value))
        UNWIND chunk_ids AS chunk_id
        RETURN chunk_id, e.norm AS norm, e.name AS name, e.label AS label
        """,
        chunk_ids=chunk_ids, database_=settings.neo4j_database,
    ).records
    for r in ent_rows:
        eid = f"entity:{r['norm']}"
        nodes.setdefault(eid, {"id": eid, "label": r["name"], "kind": "entity",
                               "source": "documents", "entityType": r["label"]})
        edges.append({"source": r["chunk_id"], "target": eid, "type": "MENTIONS"})

    bridge_rows = driver().execute_query(
        """
        MATCH (e:Entity)-[:REFERS_TO]->(v:Value)
        OPTIONAL MATCH (t:Table)-[:HAS_COLUMN]->(:Column)-[:HAS_VALUE]->(v)
        RETURN e.norm AS enorm, v.norm AS vnorm, v.name AS vname, collect(DISTINCT t.id) AS tables
        """,
        database_=settings.neo4j_database,
    ).records
    for r in bridge_rows:
        eid = f"entity:{r['enorm']}"
        if eid not in nodes:        # only bridge entities that survived the chunk cap
            continue
        vid = f"value:{r['vnorm']}"
        nodes.setdefault(vid, {"id": vid, "label": r["vname"], "kind": "value", "source": "catalog"})
        edges.append({"source": eid, "target": vid, "type": "REFERS_TO"})
        for tid in r["tables"]:
            if tid in nodes:
                edges.append({"source": vid, "target": tid, "type": "HAS_VALUE"})

    return {"nodes": list(nodes.values()), "edges": edges, "truncated": truncated}
```

- [ ] **Step 5: Run to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_graph_api_bounded.py tests/test_web_graph_api.py -v`
Expected: new tests PASS (or SKIP without Neo4j); the existing `test_web_graph_api.py` stays green — confirm it doesn't assert on the absence of a `truncated` key (the key is additive). If it ran a full DB-touching suite, restore embeddings after with `make ingest`.

- [ ] **Step 6: Commit**

```bash
git add backend/semantic_layer/config.py backend/semantic_layer/web/graph_api.py backend/tests/test_graph_api_bounded.py
git commit -m "feat(graph): bounded get_schema_graph — chunk cap + source filter + truncated"
```

---

### Task A2: Expose `source` / `max_chunks` on the `/graph` endpoint

**Files:**
- Modify: `backend/semantic_layer/web/app.py`
- Test: `backend/tests/test_graph_api_bounded.py`

**Interfaces:**
- Consumes: `get_schema_graph(source, max_chunks)` (A1).
- Produces: `GET /graph?source=<name>&max_chunks=<int>` — both optional query params forwarded to `get_schema_graph`; no params behaves as before (bounded by the default cap).

- [ ] **Step 1: Write the failing test**

```python
# append to backend/tests/test_graph_api_bounded.py
def test_graph_endpoint_forwards_query_params(monkeypatch):
    from fastapi.testclient import TestClient
    from semantic_layer.web import app as app_mod

    captured = {}
    monkeypatch.setattr(app_mod, "get_schema_graph",
                        lambda source=None, max_chunks=None: captured.update(
                            source=source, max_chunks=max_chunks) or {"nodes": [], "edges": [], "truncated": False})
    client = TestClient(app_mod.app)
    r = client.get("/graph", params={"source": "sales_pg", "max_chunks": 50})
    assert r.status_code == 200
    assert captured == {"source": "sales_pg", "max_chunks": 50}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_graph_api_bounded.py::test_graph_endpoint_forwards_query_params -v`
Expected: FAIL — the current `/graph` handler ignores query params, so `captured` stays `{"source": None, "max_chunks": None}`.

- [ ] **Step 3: Implement**

In `backend/semantic_layer/web/app.py`, change the `/graph` route to accept the params:

```python
@app.get("/graph")
def graph(source: str | None = None, max_chunks: int | None = None):
    return get_schema_graph(source=source, max_chunks=max_chunks)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_graph_api_bounded.py tests/test_web_graph_api.py -v`
Expected: PASS; existing graph endpoint test stays green.

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/web/app.py backend/tests/test_graph_api_bounded.py
git commit -m "feat(web): /graph accepts source + max_chunks query params"
```

---

# Feature B — API Federation

**Why:** Three places hardcode the four demo APIs: `pipeline.run_ingest` (`extract_all_apis(..., ("crm","itsm","partner","dgx"))`), `api_tools._SOURCES`, and `legs._API_LEG_PROMPT` (which lists every endpoint inline). Adding a 5th API is a code change, and the static endpoint prompt cannot scale to many APIs. Make the registry config-driven and feed the API leg graph-routed endpoints.

---

### Task B1: Config-driven API registry

**Files:**
- Modify: `backend/semantic_layer/config.py`
- Modify: `backend/semantic_layer/agent/api_tools.py`
- Modify: `backend/semantic_layer/ingest/pipeline.py`
- Test: `backend/tests/test_api_federation.py`

**Interfaces:**
- Produces: `settings.api_source_list -> list[str]` (parsed from `settings.api_sources`, a comma-separated string, default `"crm,itsm,partner,dgx"`). `api_tools.call_api` validates `source` against `settings.api_source_list`; `pipeline.run_ingest` registers `settings.api_source_list` instead of the hardcoded tuple.

- [ ] **Step 1: Add the setting**

In `backend/semantic_layer/config.py`, add the field (near the other source settings) and a parsed property (next to `postgres_dsn`):

```python
    # Enterprise APIs to register/serve (comma-separated). Adding an API is config, not
    # code: list its name here and mount/serve its OpenAPI spec under /{name}.
    api_sources: str = "crm,itsm,partner,dgx"
```

and the property:

```python
    @property
    def api_source_list(self) -> list[str]:
        return [s.strip() for s in self.api_sources.split(",") if s.strip()]
```

- [ ] **Step 2: Write the failing tests**

```python
# backend/tests/test_api_federation.py
import json

from semantic_layer.config import settings
from semantic_layer.agent import api_tools


def test_api_source_list_parses_csv(monkeypatch):
    monkeypatch.setattr(settings, "api_sources", "crm, itsm ,newapi", raising=False)
    assert settings.api_source_list == ["crm", "itsm", "newapi"]


def test_call_api_validates_against_config(monkeypatch):
    monkeypatch.setattr(api_tools.settings, "api_sources", "crm,itsm,partner,dgx", raising=False)
    out = json.loads(api_tools.call_api.invoke({"source": "nope", "path": "/x"}))
    assert out["status"] == 404
    assert "unknown api source" in out["error"]
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_api_federation.py -v`
Expected: FAIL — `settings.api_source_list` does not exist (AttributeError); `api_tools` has no `settings` reference yet.

- [ ] **Step 4: Implement**

In `backend/semantic_layer/agent/api_tools.py`, replace the hardcoded set with the config list:

```python
from semantic_layer.config import settings
```

and change the guard in `call_api` (remove the module-level `_SOURCES = {...}` line):

```python
    if source not in settings.api_source_list:
        return json.dumps({"status": 404, "error": f"unknown api source '{source}'"})
```

In `backend/semantic_layer/ingest/pipeline.py`, change the API registration line:

```python
        bundles += extract_all_apis(_api_spec_getter(), settings.api_source_list)
```

- [ ] **Step 5: Run to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_api_federation.py tests/test_agent_api_tools.py tests/test_api_extractor.py -v`
Expected: new tests PASS; existing API tool/extractor tests stay green (the default `api_source_list` is the same four names).

- [ ] **Step 6: Commit**

```bash
git add backend/semantic_layer/config.py backend/semantic_layer/agent/api_tools.py backend/semantic_layer/ingest/pipeline.py backend/tests/test_api_federation.py
git commit -m "feat(api): config-driven API registry (settings.api_source_list)"
```

---

### Task B2: Route API endpoints from the graph into the API leg

**Files:**
- Modify: `backend/semantic_layer/agent/routing.py`
- Modify: `backend/semantic_layer/agent/legs.py`
- Test: `backend/tests/test_api_federation.py`

**Interfaces:**
- Produces: `route_api_endpoints(intents: list[str], limit: int = 12) -> list[dict]` in `routing.py` — returns `[{"source", "path", "summary"}]` for `rest-api` endpoints whose endpoint name or summary matches any intent token (ranked by match count). `run_api_leg` (legs.py) includes these routed endpoints in its prompt instead of the static four-API list, so it scales to many registered APIs.

- [ ] **Step 1: Write the failing tests**

```python
# append to backend/tests/test_api_federation.py
import pytest

from semantic_layer.agent import routing


@pytest.mark.neo4j
def test_route_api_endpoints_finds_ticket_endpoint(ingested_graph):
    eps = routing.route_api_endpoints(["open tickets"])
    # ITSM's /tickets endpoint should surface for a ticket-related intent.
    assert any(e["source"] == "itsm" and e["path"] == "/tickets" for e in eps)
    for e in eps:
        assert set(e) >= {"source", "path", "summary"}


def test_run_api_leg_uses_routed_endpoints(monkeypatch):
    from semantic_layer.agent import legs as legs_mod
    monkeypatch.setattr(legs_mod, "route_api_endpoints",
                        lambda intents, limit=12: [{"source": "itsm", "path": "/tickets",
                                                    "summary": "List support tickets"}])
    plan_calls = legs_mod._ApiCalls(calls=[
        legs_mod._ApiCall(source="itsm", path="/tickets", params={"status": "open"})])
    captured = {}

    class _FakeStructured:
        def invoke(self, messages):
            captured["human"] = messages[-1][1]
            return plan_calls

    class _FakeModel:
        def with_structured_output(self, _schema, **_kw):
            return _FakeStructured()

    monkeypatch.setattr(legs_mod, "get_chat_model", lambda model=None: _FakeModel())
    monkeypatch.setattr(legs_mod, "call_api", type("T", (), {
        "invoke": staticmethod(lambda _a: json.dumps({"status": 200, "data": [{"id": 1}]}))})())
    out = legs_mod.run_api_leg(["open tickets"])
    assert out["error"] is None
    assert out["calls"][0]["path"] == "/tickets"
    # the routed endpoint was injected into the prompt the model saw
    assert "/tickets" in captured["human"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_api_federation.py::test_run_api_leg_uses_routed_endpoints -v`
Expected: FAIL — `routing.route_api_endpoints` / `legs.route_api_endpoints` do not exist.

- [ ] **Step 3: Implement the router**

In `backend/semantic_layer/agent/routing.py`, add:

```python
_API_EP_CYPHER = """
UNWIND $tokens AS tok
MATCH (d:Database)-[:HAS_SCHEMA]->(:Schema)-[:HAS_TABLE]->(t:Table)
WHERE toUpper(coalesce(d.platform,'')) = 'REST-API'
  AND (toLower(t.name) CONTAINS tok OR toLower(coalesce(t.description,'')) CONTAINS tok)
WITH d.name AS source, t.name AS endpoint, t.description AS summary, count(*) AS score
RETURN source, endpoint, summary, score ORDER BY score DESC, source, endpoint LIMIT $limit
"""


def route_api_endpoints(intents: list[str], limit: int = 12) -> list[dict]:
    """Retrieve the rest-api endpoints relevant to the given lookups from the graph,
    so the API leg sees only the pertinent endpoints (scales past a static prompt).
    Endpoint table names look like 'GET /tickets' -> path '/tickets'."""
    tokens = [t for intent in intents for t in intent.lower().split() if len(t) > 2]
    if not tokens:
        return []
    recs = driver().execute_query(
        _API_EP_CYPHER, tokens=tokens, limit=limit, database_=settings.neo4j_database,
    ).records
    out = []
    for r in recs:
        # endpoint name is "<METHOD> <path>"; take the path part.
        parts = r["endpoint"].split(" ", 1)
        path = parts[1] if len(parts) == 2 else r["endpoint"]
        out.append({"source": r["source"], "path": path, "summary": r["summary"] or ""})
    return out
```

- [ ] **Step 4: Feed routed endpoints into the API leg**

In `backend/semantic_layer/agent/legs.py`, import the router (near the top imports):

```python
from semantic_layer.agent.routing import route_api_endpoints
```

In `run_api_leg`, build the endpoint listing from the router and append it to the human message. Replace the body up to `plan = model.invoke(...)`:

```python
def run_api_leg(api_intents: list[str]) -> dict:
    # function_calling tolerates the open `params` dict; strict structured output rejects it.
    model = get_chat_model(settings.llm_model).with_structured_output(
        _ApiCalls, method="function_calling")
    routed = route_api_endpoints(api_intents)
    catalog = "\n".join(f"- {e['source']} {e['path']} — {e['summary']}" for e in routed) \
        or "(no matching endpoints found; use your general knowledge of the sources)"
    plan = model.invoke([
        ("system", _API_LEG_PROMPT),
        ("human", f"Lookups: {'; '.join(api_intents)}\n\nRelevant endpoints:\n{catalog}")])
```

(Leave the rest of `run_api_leg` — the per-call loop, `_normalize_params`, the result assembly — unchanged.)

- [ ] **Step 5: Run to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_api_federation.py tests/test_legs.py -v`
Expected: new tests PASS (neo4j SKIP if no DB); existing `test_legs.py` stays green (its `run_api_leg` test monkeypatches `get_chat_model` and `call_api`; if it now also needs `route_api_endpoints` monkeypatched to avoid a live graph call, add that stub — the test for `run_api_leg` should patch `legs_mod.route_api_endpoints` to return `[]`). If `test_legs.py::test_run_api_leg_executes_planned_calls` fails because it calls the real `route_api_endpoints` (which hits Neo4j), patch it in that test: `monkeypatch.setattr(legs_mod, "route_api_endpoints", lambda intents, limit=12: [])`.

- [ ] **Step 6: Commit**

```bash
git add backend/semantic_layer/agent/routing.py backend/semantic_layer/agent/legs.py backend/tests/test_api_federation.py backend/tests/test_legs.py
git commit -m "feat(api): route graph endpoints into the API leg (scales past static prompt)"
```

---

# Feature C — Schema-Routing Activation

**Why:** Schema routing (P0/P1) is built but default-OFF and unbounded in its effect: even when on, `build_plan` still resolves values and dimension targets across the whole catalog. To make "thousands of tables" tractable, bound the planner's join targets to the routed top-K when routing is on, and add an A/B eval so the default-on flip is data-backed (the eval harness exists from P1b).

---

### Task C1: Bound join targets to the routed set when routing is on

**Files:**
- Modify: `backend/semantic_layer/config.py`
- Modify: `backend/semantic_layer/agent/planner.py`
- Test: `backend/tests/test_routing_activation.py`

**Interfaces:**
- Consumes: `routed_tables` (already computed in `build_plan`), `settings.schema_routing_max_targets`.
- Produces: when `settings.schema_routing_enabled` and `routed_tables` is non-empty, `build_plan` restricts the sales-leg join targets to tables in the routed set and caps them at `settings.schema_routing_max_targets`. Routing-OFF behavior is byte-for-byte unchanged.

- [ ] **Step 1: Add the setting**

In `backend/semantic_layer/config.py`, with the other `schema_routing_*` settings:

```python
    # When routing is enabled, cap the sales-leg join targets to this many routed
    # tables so a large catalog yields a bounded join (not a whole-schema scan).
    schema_routing_max_targets: int = 8
```

- [ ] **Step 2: Write the failing tests**

```python
# backend/tests/test_routing_activation.py
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
    monkeypatch.setattr(planner_mod, "select_fact_table", lambda routed: None)


def test_routing_on_bounds_targets_to_routed_and_cap(monkeypatch):
    # dimension scan finds 3 sales tables; routing returns only 1 of them -> bounded to that 1.
    _stub_graph(monkeypatch, dim_targets=[
        "table:sales_pg.sales.segment", "table:sales_pg.sales.region",
        "table:sales_pg.sales.industry"])
    monkeypatch.setattr(planner_mod.settings, "schema_routing_enabled", True, raising=False)
    monkeypatch.setattr(planner_mod.settings, "schema_routing_max_targets", 8, raising=False)
    monkeypatch.setattr(planner_mod, "route_tables",
                        lambda q, k_ret, k_rank: ["table:sales_pg.sales.segment"])
    plan = build_plan(Intent(group_by=["segment"], needs_sql=True), question="rev by segment")
    targets = [jt["table_id"] for jt in plan["sql_legs"][0]["join_targets"]]
    assert targets == ["table:sales_pg.sales.segment"]   # bounded to the routed table


def test_routing_off_is_unchanged(monkeypatch):
    _stub_graph(monkeypatch, dim_targets=["table:sales_pg.sales.segment",
                                          "table:sales_pg.sales.region"])
    monkeypatch.setattr(planner_mod.settings, "schema_routing_enabled", False, raising=False)
    plan = build_plan(Intent(group_by=["segment", "region"], needs_sql=True))
    targets = {jt["table_id"] for jt in plan["sql_legs"][0]["join_targets"]}
    assert targets == {"table:sales_pg.sales.segment", "table:sales_pg.sales.region"}
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_routing_activation.py -v`
Expected: FAIL — `test_routing_on_bounds_targets_to_routed_and_cap` gets all 3 targets (routing currently UNIONS routed tables in, not bounds to them).

- [ ] **Step 4: Implement**

In `backend/semantic_layer/agent/planner.py`, find the sales-target assembly (the lines computing `routed_sales` and `sales_target_ids`). Replace:

```python
    routed_sales = [t for t in routed_tables if t.startswith("table:sales_pg.sales.")]
    # A sales leg is warranted by filter values, group-by dimensions, OR routed tables.
    sales_target_ids = [r["table_id"] for r in sales_dims] + dim_targets + routed_sales
```

with:

```python
    routed_sales = [t for t in routed_tables if t.startswith("table:sales_pg.sales.")]
    sales_target_ids = [r["table_id"] for r in sales_dims] + dim_targets + routed_sales
    # When routing is on, BOUND the join to the routed top-K so a large catalog yields a
    # bounded join rather than a whole-schema scan; preserve order, de-dupe, then cap.
    if settings.schema_routing_enabled and routed_sales:
        routed_set = set(routed_sales)
        bounded = [t for t in dict.fromkeys(sales_target_ids) if t in routed_set]
        sales_target_ids = bounded[: settings.schema_routing_max_targets]
```

(When routing is off, `routed_sales` is `[]`, so this block is skipped and `sales_target_ids` keeps its current value — unchanged behavior.)

- [ ] **Step 5: Run to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_routing_activation.py tests/test_planner_plan.py tests/test_planner_routing.py tests/test_routing_fact_table.py -v`
Expected: new tests PASS; existing planner/routing tests stay green (routing default-off; the one known neo4j doc-embedding failure aside).

- [ ] **Step 6: Commit**

```bash
git add backend/semantic_layer/config.py backend/semantic_layer/agent/planner.py backend/tests/test_routing_activation.py
git commit -m "feat(routing): bound sales join targets to routed top-K when enabled"
```

---

### Task C2: Routing A/B eval comparison

**Files:**
- Create: `backend/semantic_layer/eval/compare.py`
- Test: `backend/tests/test_routing_activation.py`

**Interfaces:**
- Consumes: `eval.run.run_eval`, `eval.evalset.load_evalset`, `settings.schema_routing_enabled`.
- Produces: `compare_routing(evalset, run_fn=run_eval) -> dict` → `{"off": <report>, "on": <report>, "delta_mean": float}` — runs the evalset with `schema_routing_enabled` forced False then True (restoring the original after), so you can see whether routing helps before flipping the default. `main()` CLI prints the comparison as JSON.

- [ ] **Step 1: Write the failing test**

```python
# append to backend/tests/test_routing_activation.py
from semantic_layer.eval import compare as compare_mod
from semantic_layer.eval.compare import compare_routing


def test_compare_routing_runs_both_modes_and_restores(monkeypatch):
    monkeypatch.setattr(compare_mod.settings, "schema_routing_enabled", False, raising=False)
    seen = []

    def fake_run(evalset):
        seen.append(compare_mod.settings.schema_routing_enabled)
        # mean depends on the flag so we can tell the modes apart
        return {"results": [], "mean_score": 4.0 if compare_mod.settings.schema_routing_enabled else 3.0,
                "pass_rate": 1.0}

    out = compare_routing([{"id": "x", "question": "q", "expect": "e"}], run_fn=fake_run)
    assert seen == [False, True]                       # ran OFF then ON
    assert out["off"]["mean_score"] == 3.0
    assert out["on"]["mean_score"] == 4.0
    assert out["delta_mean"] == 1.0                    # on - off
    assert compare_mod.settings.schema_routing_enabled is False   # restored
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_routing_activation.py::test_compare_routing_runs_both_modes_and_restores -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'semantic_layer.eval.compare'`.

- [ ] **Step 3: Implement**

Create `backend/semantic_layer/eval/compare.py`:

```python
"""A/B compare answer quality with schema routing OFF vs ON over the golden evalset.

Run before flipping `schema_routing_enabled` on by default: confirm ON >= OFF."""

import json

from semantic_layer.config import settings
from semantic_layer.eval.evalset import load_evalset
from semantic_layer.eval.run import run_eval


def compare_routing(evalset: list[dict], run_fn=run_eval) -> dict:
    """Run the evalset with routing forced OFF then ON; restore the original setting."""
    original = settings.schema_routing_enabled
    try:
        settings.schema_routing_enabled = False
        off = run_fn(evalset)
        settings.schema_routing_enabled = True
        on = run_fn(evalset)
    finally:
        settings.schema_routing_enabled = original
    return {"off": off, "on": on,
            "delta_mean": round(on["mean_score"] - off["mean_score"], 2)}


def main() -> None:
    print(json.dumps(compare_routing(load_evalset()), indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_routing_activation.py -v`
Expected: all routing-activation tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/eval/compare.py backend/tests/test_routing_activation.py
git commit -m "feat(eval): routing A/B comparison harness"
```

- [ ] **Step 6: (Operational, no code) Decide the default-on flip**

After merge, with services up and `OPENAI_API_KEY` set, run the comparison and record the result:
Run: `cd backend && .venv/bin/python -m semantic_layer.eval.compare`
If `delta_mean >= 0` (routing does not regress quality) and the catalog is large enough to benefit, flip `schema_routing_enabled` to `True` in `.env`/deploy config. Do NOT flip the code default blindly — this is a measured decision.

---

## Deferred to a separate plan (P2-infra) — and why

These are required for true horizontal/enterprise scale but are infrastructure/ops projects that need service provisioning and design decisions, not pure application code — keep them as a dedicated follow-up:

1. **Redis-backed shared cache** (multi-worker): a `RedisQueryCache` implementing the P0 `get_exact`/`get_semantic`/`put` interface behind `cache_backend="redis"`, with semantic lookup on a Redis vector index (RediSearch). Needs a Redis service in `docker-compose`, the `redis` dependency, and a RediSearch design — see the Caching section of the design spec.
2. **Distributed / bulk ingestion**: `neo4j-admin import` for initial multi-million-entity load + a streaming/queue pipeline (CDC) for ongoing growth; the current ingest is single-process.
3. **Horizontal Neo4j** (clustering / Infinigraph sharding) and **read replicas** — only when node/edge volume actually demands it.
4. **Auth / multi-tenancy / rate-limiting** on the agent API and the enterprise APIs (currently unauthenticated) — required before any external exposure.
5. **Entity resolution at scale**: replace token-matching `Entity -> Value` bridging with a real ER step (blocking + similarity) at millions of entities.
6. **True Yen's k-shortest join paths** via Neo4j GDS (the P1b note) — needs the GDS plugin.

---

## Self-Review

**Spec coverage** (against the P2 scale-out gaps from the assessment):
- ✅ Bound whole-graph reads (`/graph` blowup) — Feature A.
- ✅ API federation over many APIs — Feature B (config registry B1 + endpoint routing B2). Full MCP-gateway/auth deferred to P2-infra.
- ✅ Schema-routing activation for many tables — Feature C (bounded targets C1 + A/B eval C2; default-on flip is a measured operational step).
- ⏸️ Redis shared cache, distributed/bulk ingest, clustering, auth/multi-tenancy, entity resolution, GDS k-shortest — explicitly deferred to P2-infra with rationale.

**Placeholder scan:** No TBDs; every code step has complete code; every run step states the expected result and the venv command.

**Type consistency:**
- `get_schema_graph(source=None, max_chunks=None) -> {nodes, edges, truncated}` consumed by `/graph` (A1↔A2).
- `settings.api_source_list -> list[str]` consumed by `call_api` and `pipeline` (B1); `route_api_endpoints(intents, limit) -> [{source, path, summary}]` consumed by `run_api_leg` (B2).
- `build_plan` routed-target bounding reads `settings.schema_routing_enabled` + `schema_routing_max_targets` (C1); `compare_routing(evalset, run_fn) -> {off, on, delta_mean}` wraps `run_eval` (C2).

**Backward-compat:** A returns the same shape plus an additive `truncated` key, capped only past the default; B's default `api_source_list` is the same four names and `run_api_leg` falls back to a general message when no endpoints match; C's bounding only triggers when `schema_routing_enabled` (default False). The known pre-existing doc-embedding suite flake is unrelated.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-18-production-scaling-p2.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
