# Production-Scaling P0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the three highest-leverage scaling bottlenecks in the NeoCarta-Local semantic layer — schema-routing correctness at many tables, single-threaded per-chunk entity extraction, and the total absence of query caching — so the system survives 1000s of docs / thousands of tables / repeated queries.

**Architecture:** Three independent, additive features. (A) A retrieve-then-rank **schema-routing stage** (`agent/routing.py`) decouples table selection from SQL generation — the pattern LinkedIn's production text-to-SQL system used to lift correctness 9%→48% and cut schema hallucination 23%→1% (arXiv 2507.14372); also DBCopilot (EDBT 2025, arXiv 2312.03463). It is **feature-flagged off by default** so existing golden tests are untouched until enabled. (B) **Batched + parallel entity extraction** (`ingest/entities.py` + `ingest/pipeline.py`) replaces N sequential single-chunk LLM calls with M-chunk batches run concurrently. (C) A **query cache** (`agent/cache.py`) adds exact-match then semantic (embedding-cosine) lookup in front of `answer_stream`.

**Tech Stack:** Python ≥3.11, Neo4j (`neo4j` driver), LangChain `init_chat_model` (`openai:gpt-5.4-mini`), OpenAI embeddings (`text-embedding-3-small`, 1536 dims), pydantic / pydantic-settings, pytest.

## Global Constraints

- **Python:** `requires-python = ">=3.11"` (`backend/pyproject.toml`). No new runtime dependency without adding it to `pyproject.toml`.
- **Models come from config, never hardcoded:** always obtain chat models via `semantic_layer.ingest.llm.get_chat_model(<settings model id>)` and embeddings via `get_openai_client()`. Model ids live in `config.Settings` (`llm_model`, `planner_model_resolved`, `synthesis_model_resolved`, `embedding_model`, `embedding_dimensions`).
- **Tests run from `backend/`:** `cd backend && python -m pytest` (see `Makefile` `test:`). `pythonpath = ["."]`.
- **Test markers (`backend/pyproject.toml`):** `neo4j` (needs docker Neo4j), `postgres`, `openai` (needs `OPENAI_API_KEY`). Marker-gated tests must take the matching fixture (`neo4j_driver`/`ingested_graph`, `postgres_dsn`, `require_openai`) so they **skip** rather than fail when the service is absent. Pure-logic tests take none and always run.
- **LLM stubbing convention (copy from `tests/test_legs.py`):** monkeypatch the module-level `get_chat_model` with a fake returning a `_FakeModel` whose `.with_structured_output(...)` returns a `_FakeStructured` with a fixed `.invoke()` value. Never call a real model in a non-`openai` test.
- **Graph writes stay idempotent:** any new Cypher write uses `MERGE`, mirroring the rest of `ingest/`.
- **⚠️ Running the backend pytest suite wipes `Chunk.embedding` (tests reset the shared Neo4j graph).** After any run that touches the DB, restore doc RAG with `make ingest` (from repo root) or doc-search will silently return nothing. Pure-logic tasks (B1, C1 unit parts) don't trigger this; integration tasks (A1, A3, B2, C2/C3 neo4j tests) do.
- **TDD + frequent commits:** every task is test-first and ends with a commit.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `backend/semantic_layer/config.py` (modify) | New settings: routing flags, entity batching, cache | A3, B2, C1, C2 |
| `backend/semantic_layer/agent/routing.py` (create) | Retrieve candidate tables → LLM rank → top-K table ids | A1, A2 |
| `backend/semantic_layer/agent/planner.py` (modify) | Wire routing into `build_plan` behind flag | A3 |
| `backend/semantic_layer/ingest/entities.py` (modify) | `extract_entities_batch` — one call, many chunks | B1 |
| `backend/semantic_layer/ingest/pipeline.py` (modify) | Parallel batched entity extraction in `_run_llm_stages` | B2 |
| `backend/semantic_layer/agent/cache.py` (create) | `QueryCache`: exact + semantic lookup, TTL, LRU bound | C1, C2 |
| `backend/semantic_layer/agent/controller.py` (modify) | Cache lookup/store around `answer_stream` | C1, C2 |
| `backend/tests/test_routing.py` (create) | Routing unit + integration tests | A1, A2 |
| `backend/tests/test_planner_routing.py` (create) | `build_plan` routing-flag tests | A3 |
| `backend/tests/test_entities_batch.py` (create) | Batch extraction tests | B1 |
| `backend/tests/test_pipeline_entities.py` (create) | Parallel batch runner test | B2 |
| `backend/tests/test_query_cache.py` (create) | Cache unit + controller integration tests | C1, C2, C3 |

The three features are independent and may be implemented in any order. Within a feature, tasks are sequential.

---

# Feature A — Schema-Routing Stage (retrieve → rank → constrain)

**Why:** `build_plan` currently hardcodes the fact table (`_SALES_FACT = "table:sales_pg.sales.order_line"`) and maps group-by tokens to sales tables by name. At thousands of tables this both misses the right tables and, if schemas are fed to the SQL LLM wholesale, degrades comprehension (DBCopilot, arXiv 2312.03463). This feature adds a reusable router and wires it in **behind `settings.schema_routing_enabled` (default `False`)** so current behavior and golden tests are preserved until the flag is turned on.

---

### Task A1: Candidate-table retrieval (high recall)

**Files:**
- Create: `backend/semantic_layer/agent/routing.py`
- Test: `backend/tests/test_routing.py`

**Interfaces:**
- Consumes: `search_catalog` (`agent/graph_tools.py`, returns JSON list of `{kind, id, name, table_id, ...}`), `driver()` (`agent/driver.py`), `settings` (`config.py`).
- Produces: `retrieve_candidate_tables(question: str, k_ret: int = 20) -> list[dict]` returning deduped `[{"table_id": str, "score": int}]` ordered by `score` DESC, length ≤ `k_ret`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_routing.py
import json
import pytest

from semantic_layer.agent import routing


def test_retrieve_candidate_tables_dedups_and_ranks(monkeypatch):
    # search_catalog is a LangChain tool invoked via .invoke({"query": ...}).
    hits = [
        {"kind": "value", "id": "c1", "name": "EMEA", "table_id": "table:sales_pg.sales.region", "score": 3},
        {"kind": "column", "id": "c2", "name": "segment", "table_id": "table:sales_pg.sales.segment", "score": 2},
        {"kind": "value", "id": "c3", "name": "EMEA", "table_id": "table:sales_pg.sales.region", "score": 1},
    ]
    monkeypatch.setattr(routing, "search_catalog", type("T", (), {
        "invoke": staticmethod(lambda _a: json.dumps(hits))})())
    out = routing.retrieve_candidate_tables("revenue by segment in EMEA", k_ret=20)
    ids = [c["table_id"] for c in out]
    assert ids == ["table:sales_pg.sales.region", "table:sales_pg.sales.segment"]  # deduped, score-summed, ranked
    assert out[0]["score"] == 4  # 3 + 1 for region
    assert len(out) <= 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_routing.py::test_retrieve_candidate_tables_dedups_and_ranks -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'semantic_layer.agent.routing'`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/semantic_layer/agent/routing.py
"""Schema routing: retrieve a high-recall set of candidate tables for a question,
then rank them with one LLM call. Decouples table selection from SQL generation
so text-to-SQL scales past a handful of tables (cf. DBCopilot, arXiv 2312.03463;
LinkedIn text-to-SQL, arXiv 2507.14372)."""

import json

from semantic_layer.agent.graph_tools import search_catalog


def retrieve_candidate_tables(question: str, k_ret: int = 20) -> list[dict]:
    """High-recall candidate tables for a question.

    Aggregates search_catalog hits (keyword/value/business-term) by their owning
    table, summing hit scores. Returns [{table_id, score}] ranked DESC, capped at
    k_ret. Tuned for recall: k_ret is intentionally generous; the LLM ranker
    (rank_tables) trims to a precise set."""
    hits = json.loads(search_catalog.invoke({"query": question}))
    scores: dict[str, int] = {}
    for h in hits:
        tid = h.get("table_id")
        if not tid or not tid.startswith("table:"):
            continue
        scores[tid] = scores.get(tid, 0) + int(h.get("score") or 1)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"table_id": tid, "score": s} for tid, s in ranked[:k_ret]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_routing.py::test_retrieve_candidate_tables_dedups_and_ranks -v`
Expected: PASS.

- [ ] **Step 5: Add a neo4j integration test (skips when DB absent)**

```python
# append to backend/tests/test_routing.py
@pytest.mark.neo4j
def test_retrieve_candidate_tables_finds_real_dimensions(ingested_graph):
    out = routing.retrieve_candidate_tables("Data Center revenue from Cloud customers in EMEA", k_ret=20)
    ids = {c["table_id"] for c in out}
    # 'EMEA' is a region value, 'Cloud' an industry value, 'Data Center' a segment value.
    assert "table:sales_pg.sales.region" in ids
    assert any(t.endswith("industry") or t.endswith("segment") for t in ids)
```

- [ ] **Step 6: Run both tests**

Run: `cd backend && python -m pytest tests/test_routing.py -v`
Expected: unit PASS; neo4j test PASS (or SKIP if Neo4j unavailable).

- [ ] **Step 7: Commit**

```bash
git add backend/semantic_layer/agent/routing.py backend/tests/test_routing.py
git commit -m "feat(routing): high-recall candidate-table retrieval"
```

---

### Task A2: LLM table ranker (precision)

**Files:**
- Modify: `backend/semantic_layer/agent/routing.py`
- Test: `backend/tests/test_routing.py`

**Interfaces:**
- Consumes: `get_chat_model` (`ingest/llm.py`), `settings.planner_model_resolved`, `retrieve_candidate_tables` (Task A1).
- Produces:
  - `rank_tables(question: str, candidates: list[dict], k_rank: int = 8, min_score: int = 3) -> list[str]` — one structured LLM call; returns ≤ `k_rank` `table_id`s scored ≥ `min_score`, ranked DESC.
  - `route_tables(question: str, k_ret: int = 20, k_rank: int = 8) -> list[str]` — composes retrieve + rank; returns `[]` when there are no candidates.

- [ ] **Step 1: Write the failing test**

```python
# append to backend/tests/test_routing.py
from semantic_layer.agent import routing as routing_mod


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


def test_rank_tables_filters_by_threshold_and_limit(monkeypatch):
    candidates = [
        {"table_id": "table:sales_pg.sales.segment", "score": 5},
        {"table_id": "table:sales_pg.sales.region", "score": 4},
        {"table_id": "table:sales_pg.sales.industry", "score": 1},
    ]
    scores = routing_mod._TableScores(scores=[
        routing_mod._TableScore(table_id="table:sales_pg.sales.segment", score=5),
        routing_mod._TableScore(table_id="table:sales_pg.sales.region", score=4),
        routing_mod._TableScore(table_id="table:sales_pg.sales.industry", score=1),  # below min_score
    ])
    monkeypatch.setattr(routing_mod, "get_chat_model", lambda model=None: _FakeModel(scores))
    out = routing_mod.rank_tables("revenue by segment in EMEA", candidates, k_rank=8, min_score=3)
    assert out == ["table:sales_pg.sales.segment", "table:sales_pg.sales.region"]


def test_route_tables_returns_empty_without_candidates(monkeypatch):
    monkeypatch.setattr(routing_mod, "retrieve_candidate_tables", lambda q, k_ret=20: [])
    assert routing_mod.route_tables("nonsense xyzzy") == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && python -m pytest tests/test_routing.py::test_rank_tables_filters_by_threshold_and_limit -v`
Expected: FAIL with `AttributeError: module ... has no attribute '_TableScores'`.

- [ ] **Step 3: Implement**

```python
# add to backend/semantic_layer/agent/routing.py (imports at top)
from pydantic import BaseModel, Field

from semantic_layer.config import settings
from semantic_layer.ingest.llm import get_chat_model


class _TableScore(BaseModel):
    table_id: str
    score: int = Field(ge=0, le=5)


class _TableScores(BaseModel):
    scores: list[_TableScore] = Field(default_factory=list)


_RANK_PROMPT = (
    "You route a business question to the database tables needed to answer it. "
    "For EACH candidate table id, score 0-5 how likely it is required: 5 = certainly "
    "needed (holds the measure or a filter dimension), 0 = irrelevant. Only score the "
    "candidates given; do not invent table ids. Return the list of {table_id, score}."
)


def rank_tables(question: str, candidates: list[dict], k_rank: int = 8,
                min_score: int = 3) -> list[str]:
    """One structured LLM call scoring each candidate table's relevance; returns the
    top k_rank table ids scoring >= min_score, ranked by score DESC."""
    if not candidates:
        return []
    model = get_chat_model(settings.planner_model_resolved).with_structured_output(_TableScores)
    listing = "\n".join(f"- {c['table_id']}" for c in candidates)
    result = model.invoke([
        ("system", _RANK_PROMPT),
        ("human", f"Question: {question}\n\nCandidate tables:\n{listing}"),
    ])
    kept = sorted(
        (s for s in result.scores if s.score >= min_score),
        key=lambda s: (-s.score, s.table_id),
    )
    return [s.table_id for s in kept[:k_rank]]


def route_tables(question: str, k_ret: int = 20, k_rank: int = 8) -> list[str]:
    """Retrieve high-recall candidates, then LLM-rank to a precise ordered set."""
    candidates = retrieve_candidate_tables(question, k_ret=k_ret)
    return rank_tables(question, candidates, k_rank=k_rank)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && python -m pytest tests/test_routing.py -v`
Expected: all PASS (neo4j test SKIP if no DB).

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/agent/routing.py backend/tests/test_routing.py
git commit -m "feat(routing): LLM table ranker + route_tables composition"
```

---

### Task A3: Wire routing into `build_plan` behind a flag

**Files:**
- Modify: `backend/semantic_layer/config.py` (add settings)
- Modify: `backend/semantic_layer/agent/planner.py`
- Test: `backend/tests/test_planner_routing.py`

**Interfaces:**
- Consumes: `route_tables` (Task A2), existing `_dimension_targets`, `_resolve_values`, `_SALES_FACT` (`planner.py`).
- Produces: `build_plan` returns an extra key `"routed_tables": list[str]` (empty when routing disabled or no candidates). When `settings.schema_routing_enabled` is `True`, routed sales-schema tables are unioned into the sales leg's `join_targets` and surfaced in `highlight`. Default-off path is byte-for-byte unchanged.

- [ ] **Step 1: Add settings (no test needed yet — covered by Step 2)**

In `backend/semantic_layer/config.py`, after `agent_recursion_limit` (line 39):

```python
    # Schema routing: retrieve-then-rank table selection for many-table scale.
    # Off by default so the deterministic sales planner and golden tests are
    # unchanged; turn on once the catalog grows past a handful of tables.
    schema_routing_enabled: bool = False
    schema_routing_k_ret: int = 20
    schema_routing_k_rank: int = 8
```

- [ ] **Step 2: Write the failing tests**

```python
# backend/tests/test_planner_routing.py
from semantic_layer.agent import planner as planner_mod
from semantic_layer.agent.planner import Intent, build_plan


def _stub_graph(monkeypatch, *, resolved=None, dim_targets=None, columns=None):
    """Stub the planner's graph reads so build_plan runs without Neo4j."""
    monkeypatch.setattr(planner_mod, "_resolve_values", lambda terms: resolved or [])
    monkeypatch.setattr(planner_mod, "_dimension_targets", lambda gb: dim_targets or [])
    monkeypatch.setattr(planner_mod, "_join_targets", lambda fact, ids: [
        {"table_id": t, "tables": [fact, t], "joins": []} for t in dict.fromkeys(ids)])
    monkeypatch.setattr(planner_mod, "_table_columns", lambda tid: columns or ["amount"])
    monkeypatch.setattr(planner_mod, "_context_docs", lambda terms: None)
    monkeypatch.setattr(planner_mod, "_api_correlations", lambda: [])


def test_build_plan_routing_disabled_is_unchanged(monkeypatch):
    _stub_graph(monkeypatch, dim_targets=["table:sales_pg.sales.segment"])
    monkeypatch.setattr(planner_mod.settings, "schema_routing_enabled", False, raising=False)
    called = {"routed": False}
    monkeypatch.setattr(planner_mod, "route_tables",
                        lambda *a, **k: called.__setitem__("routed", True) or [])
    plan = build_plan(Intent(group_by=["segment"], needs_sql=True))
    assert called["routed"] is False           # router not invoked when disabled
    assert plan["routed_tables"] == []
    targets = [jt["table_id"] for jt in plan["sql_legs"][0]["join_targets"]]
    assert "table:sales_pg.sales.segment" in targets


def test_build_plan_routing_enabled_unions_routed_tables(monkeypatch):
    _stub_graph(monkeypatch, dim_targets=["table:sales_pg.sales.segment"])
    monkeypatch.setattr(planner_mod.settings, "schema_routing_enabled", True, raising=False)
    monkeypatch.setattr(planner_mod, "route_tables",
                        lambda q, k_ret, k_rank: ["table:sales_pg.sales.region"])
    plan = build_plan(Intent(group_by=["segment"], needs_sql=True),
                      question="revenue by segment in EMEA")
    assert plan["routed_tables"] == ["table:sales_pg.sales.region"]
    targets = [jt["table_id"] for jt in plan["sql_legs"][0]["join_targets"]]
    assert "table:sales_pg.sales.region" in targets    # routed table folded into the join
    assert "table:sales_pg.sales.region" in plan["highlight"]
```

- [ ] **Step 3: Run to verify they fail**

Run: `cd backend && python -m pytest tests/test_planner_routing.py -v`
Expected: FAIL — `build_plan() got an unexpected keyword argument 'question'` / missing `routed_tables`.

- [ ] **Step 4: Implement**

In `backend/semantic_layer/agent/planner.py`, add the import near the existing imports (after line 16):

```python
from semantic_layer.agent.routing import route_tables
```

Change the `build_plan` signature (line 153) and body so routing is consulted only when enabled. Replace lines 153–176 (`def build_plan...` through the `sales_target_ids = ...` assignment) with:

```python
def build_plan(intent: "Intent", question: str | None = None) -> dict:
    """Deterministic graph planning. No LLM unless schema routing is enabled.
    Returns a JSON-serializable Plan dict."""
    resolved = _resolve_values(intent.terms)

    # Optional retrieve-then-rank routing: only sales-schema tables are foldable
    # into the deterministic sales leg here; other routed ids are still surfaced.
    routed_tables: list[str] = []
    if settings.schema_routing_enabled and question:
        routed_tables = route_tables(
            question, k_ret=settings.schema_routing_k_ret,
            k_rank=settings.schema_routing_k_rank)

    # Document context first, so a question with no explicit period can scope SQL to the
    # period the cited press release reports ("compare with the latest release").
    doc_leg = None
    if intent.needs_doc:
        ctx = _context_docs(intent.terms)
        if ctx is not None:
            doc_leg = {"doc_query": intent.doc_query, **ctx}

    fiscal_year, quarter = intent.fiscal_year, intent.quarter
    if fiscal_year is None and doc_leg and doc_leg.get("periods"):
        m = re.match(r"FY(\d{4})-(Q\d)", sorted(doc_leg["periods"])[-1])
        if m:
            fiscal_year, quarter = int(m.group(1)), m.group(2)
    scope = {"fiscal_year": fiscal_year, "quarter": quarter}

    sql_legs = []
    sales_dims = [r for r in resolved if r["source"] == "sales_pg"]
    dim_targets = _dimension_targets(intent.group_by) if intent.needs_sql else []
    routed_sales = [t for t in routed_tables if t.startswith("table:sales_pg.sales.")]
    # A sales leg is warranted by filter values, group-by dimensions, OR routed tables.
    sales_target_ids = [r["table_id"] for r in sales_dims] + dim_targets + routed_sales
```

Then update the `highlight` set (currently lines 199–203) to include routed tables:

```python
    highlight = sorted({
        *(r["table_id"] for r in resolved),
        *(t for leg in sql_legs for jt in leg["join_targets"] for t in jt["tables"]),
        *(doc_leg["candidate_doc_ids"] if doc_leg else []),
        *routed_tables,
    })
```

And add `routed_tables` to the returned dict (currently lines 205–211):

```python
    return {
        "resolved_values": resolved,
        "sql_legs": sql_legs,
        "doc_leg": doc_leg,
        "api_correlations": api_correlations,
        "highlight": highlight,
        "routed_tables": routed_tables,
    }
```

- [ ] **Step 5: Pass the question through from the controller**

In `backend/semantic_layer/agent/controller.py`, line 56, change:

```python
        plan = build_plan(intent)
```
to:
```python
        plan = build_plan(intent, question=question)
```

- [ ] **Step 6: Run the new tests + the existing planner/controller tests**

Run: `cd backend && python -m pytest tests/test_planner_routing.py tests/test_planner_plan.py tests/test_controller.py -v`
Expected: all PASS (existing tests still green because routing is default-off and `routed_tables` defaults to `[]`).

- [ ] **Step 7: Commit**

```bash
git add backend/semantic_layer/config.py backend/semantic_layer/agent/planner.py backend/semantic_layer/agent/controller.py backend/tests/test_planner_routing.py
git commit -m "feat(routing): wire retrieve-then-rank into build_plan behind flag"
```

---

# Feature B — Batched + Parallel Entity Extraction

**Why:** `pipeline._run_llm_stages` calls `extract_entities(row["text"])` once per chunk, sequentially (`pipeline.py:95-96`). At 1000 docs ≈ 50k chunks that is 50k serial LLM round-trips (hours). Batch many chunks per call and run batches concurrently.

---

### Task B1: `extract_entities_batch` — one call, many chunks

**Files:**
- Modify: `backend/semantic_layer/ingest/entities.py`
- Test: `backend/tests/test_entities_batch.py`

**Interfaces:**
- Consumes: `get_chat_model` (`ingest/llm.py`), `POLE_LABELS` (existing in `entities.py`).
- Produces: `extract_entities_batch(texts: list[str]) -> list[list[dict]]` — returns one entity list per input text, **same length and order as `texts`**. Each entity is `{"name": str, "label": str}`, deduped per chunk, labels constrained to `POLE_LABELS`. On parse failure or count mismatch, returns `[[] for _ in texts]`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_entities_batch.py
import json

from semantic_layer.ingest import entities as ent_mod
from semantic_layer.ingest.entities import extract_entities_batch


class _Resp:
    def __init__(self, content):
        self.content = content


class _FakeModel:
    def __init__(self, content):
        self._content = content

    def invoke(self, _prompt):
        return _Resp(self._content)


def test_extract_entities_batch_groups_per_chunk(monkeypatch):
    payload = json.dumps([
        [{"name": "NVIDIA", "label": "Org"}, {"name": "nvidia", "label": "Org"}],  # dup dropped
        [{"name": "Jensen Huang", "label": "Person"}, {"name": "Bob", "label": "Alien"}],  # bad label dropped
    ])
    monkeypatch.setattr(ent_mod, "get_chat_model", lambda model=None: _FakeModel(payload))
    out = extract_entities_batch(["chunk about NVIDIA", "chunk about the CEO"])
    assert len(out) == 2
    assert out[0] == [{"name": "NVIDIA", "label": "Org"}]
    assert out[1] == [{"name": "Jensen Huang", "label": "Person"}]


def test_extract_entities_batch_handles_bad_json(monkeypatch):
    monkeypatch.setattr(ent_mod, "get_chat_model", lambda model=None: _FakeModel("not json"))
    out = extract_entities_batch(["a", "b", "c"])
    assert out == [[], [], []]


def test_extract_entities_batch_empty_input(monkeypatch):
    monkeypatch.setattr(ent_mod, "get_chat_model",
                        lambda model=None: (_ for _ in ()).throw(AssertionError("should not call model")))
    assert extract_entities_batch([]) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && python -m pytest tests/test_entities_batch.py -v`
Expected: FAIL — `ImportError: cannot import name 'extract_entities_batch'`.

- [ ] **Step 3: Implement**

Append to `backend/semantic_layer/ingest/entities.py`:

```python
_BATCH_PROMPT = (
    "Extract named entities from EACH numbered text below. Return ONLY a JSON array "
    "with one element per text, in the same order. Each element is an array of objects "
    'with keys "name" and "label"; label must be one of: Person, Org, Location, Event, '
    "Object. Deduplicate by name within each text. Use an empty array for a text with no "
    "entities. Return exactly {n} elements.\n\n{body}"
)


def _clean_entities(raw: list) -> list[dict]:
    out, seen = [], set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip()
        label = (item.get("label") or "").strip().capitalize()
        if name and label in POLE_LABELS and name.lower() not in seen:
            seen.add(name.lower())
            out.append({"name": name, "label": label})
    return out


def extract_entities_batch(texts: list[str]) -> list[list[dict]]:
    """Extract POLE+O entities for many chunks in ONE LLM call.

    Returns one entity list per input text, in the same order. On any parse error
    or element-count mismatch, returns empty lists for every text (the caller can
    retry the batch smaller or fall back to per-chunk extraction)."""
    if not texts:
        return []
    body = "\n\n".join(f"[{i}] {t[:6000]}" for i, t in enumerate(texts))
    model = get_chat_model()
    resp = model.invoke(_BATCH_PROMPT.format(n=len(texts), body=body))
    content = resp.content if hasattr(resp, "content") else str(resp)
    content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        groups = json.loads(content)
    except json.JSONDecodeError:
        return [[] for _ in texts]
    if not isinstance(groups, list) or len(groups) != len(texts):
        return [[] for _ in texts]
    return [_clean_entities(g) for g in groups]
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && python -m pytest tests/test_entities_batch.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/ingest/entities.py backend/tests/test_entities_batch.py
git commit -m "feat(ingest): batched entity extraction (one call, many chunks)"
```

---

### Task B2: Parallel batched extraction in the pipeline

**Files:**
- Modify: `backend/semantic_layer/config.py` (add settings)
- Modify: `backend/semantic_layer/ingest/pipeline.py`
- Test: `backend/tests/test_pipeline_entities.py`

**Interfaces:**
- Consumes: `extract_entities_batch` (Task B1), `settings.entity_batch_size`, `settings.ingest_max_workers`.
- Produces: `extract_entities_for_chunks(chunk_rows: list[dict]) -> dict[str, list[dict]]` in `pipeline.py` — maps `chunk_id -> entities`, splitting rows into `entity_batch_size` batches run concurrently across `ingest_max_workers` threads. `_run_llm_stages` uses it instead of the per-chunk loop.

- [ ] **Step 1: Add settings**

In `backend/semantic_layer/config.py`, after the routing settings from Task A3 (or after `agent_recursion_limit` if A3 not done):

```python
    # Entity-extraction throughput: chunks per LLM call, and concurrent in-flight
    # batches during ingest (replaces the old one-call-per-chunk serial loop).
    entity_batch_size: int = 10
    ingest_max_workers: int = 8
```

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/test_pipeline_entities.py
from semantic_layer.ingest import pipeline as pipe


def test_extract_entities_for_chunks_covers_all_rows(monkeypatch):
    rows = [{"id": f"c{i}", "text": f"text {i}"} for i in range(25)]
    seen_batches = []

    def fake_batch(texts):
        seen_batches.append(len(texts))
        # one fake entity per chunk so we can assert mapping
        return [[{"name": t, "label": "Org"}] for t in texts]

    monkeypatch.setattr(pipe.settings, "entity_batch_size", 10, raising=False)
    monkeypatch.setattr(pipe.settings, "ingest_max_workers", 4, raising=False)
    monkeypatch.setattr(pipe, "extract_entities_batch", fake_batch)

    result = pipe.extract_entities_for_chunks(rows)
    assert set(result) == {f"c{i}" for i in range(25)}           # every chunk mapped
    assert result["c7"] == [{"name": "text 7", "label": "Org"}]  # correct row->entities
    assert sorted(seen_batches) == [5, 10, 10]                   # 25 rows -> 10+10+5
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd backend && python -m pytest tests/test_pipeline_entities.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'extract_entities_for_chunks'`.

- [ ] **Step 4: Implement**

In `backend/semantic_layer/ingest/pipeline.py`, add imports at the top of the file (after the existing imports, near line 23):

```python
from concurrent.futures import ThreadPoolExecutor

from semantic_layer.ingest.entities import extract_entities_batch
```

Add the helper (place it above `_run_llm_stages`):

```python
def extract_entities_for_chunks(chunk_rows: list[dict]) -> dict[str, list[dict]]:
    """Map chunk id -> entities, running entity_batch_size-sized batches concurrently.

    Replaces the previous one-LLM-call-per-chunk serial loop. Each batch is one LLM
    call; batches run across ingest_max_workers threads."""
    size = max(1, settings.entity_batch_size)
    batches = [chunk_rows[i:i + size] for i in range(0, len(chunk_rows), size)]
    if not batches:
        return {}

    def run(batch: list[dict]) -> dict[str, list[dict]]:
        groups = extract_entities_batch([r["text"] for r in batch])
        return {r["id"]: ents for r, ents in zip(batch, groups)}

    out: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=settings.ingest_max_workers) as pool:
        for partial in pool.map(run, batches):
            out.update(partial)
    return out
```

Then in `_run_llm_stages` (lines 91–98) replace the per-chunk loop:

```python
    with driver.session(database=settings.neo4j_database) as session:
        chunk_rows = session.run(
            "MATCH (c:Chunk) RETURN c.id AS id, c.text AS text ORDER BY c.id"
        ).data()
    entities_by_chunk = extract_entities_for_chunks(chunk_rows)
    for chunk_id, ents in entities_by_chunk.items():
        load_entities(driver, chunk_id, ents)
    # Bridge document entities to the canonical value layer (Entity -> Value).
    bridge_entities_to_values(driver)
```

(Remove the now-unused `from semantic_layer.ingest.entities import extract_entities` line inside `_run_llm_stages` at line 79.)

- [ ] **Step 5: Run the new test + the existing pipeline test**

Run: `cd backend && python -m pytest tests/test_pipeline_entities.py tests/test_pipeline.py -v`
Expected: new test PASS; `test_pipeline.py` PASS (or SKIP if it is neo4j/openai-marked and services absent).

- [ ] **Step 6: Restore embeddings if the suite touched Neo4j**

If `test_pipeline.py` ran against a live DB, run from the repo root:
```bash
make ingest
```
Expected: ingest completes; `Chunk.embedding` repopulated.

- [ ] **Step 7: Commit**

```bash
git add backend/semantic_layer/config.py backend/semantic_layer/ingest/pipeline.py backend/tests/test_pipeline_entities.py
git commit -m "feat(ingest): parallel batched entity extraction in pipeline"
```

---

# Feature C — Query Cache (exact + semantic)

**Why:** `answer_stream` re-runs intent extraction, all legs, and synthesis for every question, even an identical repeat — the agent's own estimate is ~$0.40/question. Add an exact-match then semantic (embedding-cosine) cache in front of it. In-process LRU with TTL for P0; a Redis backend is a documented follow-up (Task C3) for multi-worker deploys.

---

### Task C1: In-process cache + exact-match wiring

**Files:**
- Modify: `backend/semantic_layer/config.py` (add settings)
- Create: `backend/semantic_layer/agent/cache.py`
- Modify: `backend/semantic_layer/agent/controller.py`
- Test: `backend/tests/test_query_cache.py`

**Interfaces:**
- Consumes: `settings` (`config.py`), `time.monotonic`.
- Produces:
  - `QueryCache(max_entries: int, ttl_seconds: int)` with `get_exact(question: str) -> dict | None`, `put(question: str, answer: dict, embedding: list[float] | None = None) -> None`, and `_normalize(question) -> str` (lowercased, whitespace-collapsed). LRU-bounded, TTL-expiring.
  - Module singleton `query_cache` built from settings.
  - `answer_stream` checks the cache first and stores the final answer event payload.

- [ ] **Step 1: Add settings**

In `backend/semantic_layer/config.py`, after the entity settings from Task B2:

```python
    # Query cache: exact-match + semantic (embedding cosine) lookup in front of the
    # agent, to skip re-running legs/synthesis for repeated or near-identical asks.
    query_cache_enabled: bool = True
    cache_max_entries: int = 512
    cache_ttl_seconds: int = 3600
    cache_similarity_threshold: float = 0.95
```

- [ ] **Step 2: Write the failing unit tests**

```python
# backend/tests/test_query_cache.py
from semantic_layer.agent.cache import QueryCache


def test_exact_hit_after_put():
    c = QueryCache(max_entries=10, ttl_seconds=1000)
    c.put("What is revenue?", {"content": "42"})
    assert c.get_exact("  what   IS revenue? ") == {"content": "42"}   # normalized match


def test_exact_miss_returns_none():
    c = QueryCache(max_entries=10, ttl_seconds=1000)
    assert c.get_exact("anything") is None


def test_lru_eviction():
    c = QueryCache(max_entries=2, ttl_seconds=1000)
    c.put("a", {"content": "A"})
    c.put("b", {"content": "B"})
    c.put("c", {"content": "C"})            # evicts "a" (oldest)
    assert c.get_exact("a") is None
    assert c.get_exact("b") == {"content": "B"}
    assert c.get_exact("c") == {"content": "C"}


def test_ttl_expiry():
    clock = {"t": 1000.0}
    c = QueryCache(max_entries=10, ttl_seconds=5, now=lambda: clock["t"])
    c.put("q", {"content": "X"})
    clock["t"] = 1004.0
    assert c.get_exact("q") == {"content": "X"}   # within TTL
    clock["t"] = 1006.0
    assert c.get_exact("q") is None               # expired
```

- [ ] **Step 3: Run to verify they fail**

Run: `cd backend && python -m pytest tests/test_query_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'semantic_layer.agent.cache'`.

- [ ] **Step 4: Implement the cache**

```python
# backend/semantic_layer/agent/cache.py
"""In-process query cache: exact-match (normalized question) plus semantic
(embedding cosine) lookup, LRU-bounded with TTL. A Redis-backed variant (Task C3)
is the production path for multi-worker deploys; this default suits a single worker."""

import time
from collections import OrderedDict

from semantic_layer.config import settings


def _normalize(question: str) -> str:
    return " ".join((question or "").lower().split())


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class QueryCache:
    def __init__(self, max_entries: int, ttl_seconds: int, now=time.monotonic):
        self._max = max_entries
        self._ttl = ttl_seconds
        self._now = now
        # key -> {"answer": dict, "embedding": list[float] | None, "ts": float}
        self._store: "OrderedDict[str, dict]" = OrderedDict()

    def _fresh(self, entry: dict) -> bool:
        return (self._now() - entry["ts"]) <= self._ttl

    def get_exact(self, question: str) -> dict | None:
        key = _normalize(question)
        entry = self._store.get(key)
        if entry is None:
            return None
        if not self._fresh(entry):
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return entry["answer"]

    def get_semantic(self, embedding: list[float], threshold: float) -> dict | None:
        best, best_sim = None, threshold
        for key, entry in list(self._store.items()):
            if not self._fresh(entry):
                self._store.pop(key, None)
                continue
            emb = entry.get("embedding")
            if emb is None:
                continue
            sim = cosine(embedding, emb)
            if sim >= best_sim:
                best, best_sim = entry, sim
        return best["answer"] if best else None

    def put(self, question: str, answer: dict, embedding: list[float] | None = None) -> None:
        key = _normalize(question)
        self._store[key] = {"answer": answer, "embedding": embedding, "ts": self._now()}
        self._store.move_to_end(key)
        while len(self._store) > self._max:
            self._store.popitem(last=False)   # evict oldest


query_cache = QueryCache(settings.cache_max_entries, settings.cache_ttl_seconds)
```

- [ ] **Step 5: Run unit tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_query_cache.py -v`
Expected: PASS.

- [ ] **Step 6: Wire exact-match into `answer_stream` (failing test first)**

Add to `backend/tests/test_query_cache.py`:

```python
from semantic_layer.agent import controller as ctrl
from semantic_layer.agent import cache as cache_mod
from semantic_layer.agent.planner import Intent


def test_answer_stream_serves_exact_cache_hit(monkeypatch):
    fresh = cache_mod.QueryCache(max_entries=10, ttl_seconds=1000)
    monkeypatch.setattr(ctrl, "query_cache", fresh)
    monkeypatch.setattr(ctrl.settings, "query_cache_enabled", True, raising=False)
    calls = {"intent": 0}
    monkeypatch.setattr(ctrl, "extract_intent",
                        lambda q: calls.__setitem__("intent", calls["intent"] + 1) or Intent())
    monkeypatch.setattr(ctrl, "build_plan", lambda intent, question=None: {
        "highlight": [], "sql_legs": [], "doc_leg": None, "api_correlations": []})
    monkeypatch.setattr(ctrl, "_synthesize", lambda *a, **k: "cached-me")
    monkeypatch.setattr(ctrl, "check_numeric_grounding", lambda *a, **k: [])

    first = list(ctrl.answer_stream("Total revenue?"))
    assert first[-1]["content"] == "cached-me"
    assert calls["intent"] == 1

    second = list(ctrl.answer_stream("  total   revenue? "))   # normalized same question
    assert second[-1]["type"] == "answer"
    assert second[-1]["content"] == "cached-me"
    assert calls["intent"] == 1                                # legs NOT re-run on hit
```

- [ ] **Step 7: Run to verify it fails**

Run: `cd backend && python -m pytest tests/test_query_cache.py::test_answer_stream_serves_exact_cache_hit -v`
Expected: FAIL — `calls["intent"] == 2` (no cache yet).

- [ ] **Step 8: Implement the wiring**

In `backend/semantic_layer/agent/controller.py`, add the import (after line 14):

```python
from semantic_layer.agent.cache import query_cache
```

At the start of `answer_stream` (right after `def answer_stream(question: str) -> Iterator[dict]:`, before the `try:`):

```python
    if settings.query_cache_enabled:
        cached = query_cache.get_exact(question)
        if cached is not None:
            yield {**cached, "cached": True}
            return
```

Then, just before the final `yield {"type": "answer", ...}` (line 102), build the payload once, store it, and yield it:

```python
    answer_event = {"type": "answer", "content": summary, "highlight": plan.get("highlight", []),
                    "sql_runs": sql_runs, "api_calls": api_calls,
                    "doc_citations": doc_citations, "caveats": caveats}
    if settings.query_cache_enabled:
        query_cache.put(question, answer_event)
    yield answer_event
```

(Replace the existing final `yield {"type": "answer", ...}` statement with the three lines above.)

- [ ] **Step 9: Run the wiring test + existing controller tests**

Run: `cd backend && python -m pytest tests/test_query_cache.py tests/test_controller.py -v`
Expected: all PASS. (`test_controller.py` still green: it builds a fresh intent because its questions differ and, if needed, monkeypatch `query_cache_enabled` is not required since each test question is unique — confirm it passes; if a shared-cache interaction appears, set `monkeypatch.setattr(ctrl.settings, "query_cache_enabled", False, raising=False)` in that test.)

- [ ] **Step 10: Commit**

```bash
git add backend/semantic_layer/config.py backend/semantic_layer/agent/cache.py backend/semantic_layer/agent/controller.py backend/tests/test_query_cache.py
git commit -m "feat(cache): in-process exact-match query cache"
```

---

### Task C2: Semantic cache layer (embedding cosine)

**Files:**
- Modify: `backend/semantic_layer/agent/cache.py`
- Modify: `backend/semantic_layer/agent/controller.py`
- Test: `backend/tests/test_query_cache.py`

**Interfaces:**
- Consumes: `get_openai_client` (`ingest/llm.py`), `settings.embedding_model`, `settings.embedding_dimensions`, `settings.cache_similarity_threshold`, `QueryCache.get_semantic`/`put` (Task C1).
- Produces: `embed_query(question: str) -> list[float]` in `cache.py`; `answer_stream` falls back to semantic lookup when exact misses, and stores the question embedding on `put`.

- [ ] **Step 1: Write the failing test**

```python
# append to backend/tests/test_query_cache.py
def test_answer_stream_serves_semantic_hit(monkeypatch):
    fresh = cache_mod.QueryCache(max_entries=10, ttl_seconds=1000)
    monkeypatch.setattr(ctrl, "query_cache", fresh)
    monkeypatch.setattr(ctrl.settings, "query_cache_enabled", True, raising=False)
    monkeypatch.setattr(ctrl.settings, "cache_similarity_threshold", 0.9, raising=False)
    # Deterministic embeddings: near-identical vectors for paraphrases.
    embeds = {
        "what was total revenue": [1.0, 0.0, 0.0],
        "what is the total revenue": [0.98, 0.02, 0.0],
    }
    monkeypatch.setattr(ctrl, "embed_query", lambda q: embeds[cache_mod._normalize(q)])
    monkeypatch.setattr(ctrl, "extract_intent", lambda q: Intent())
    monkeypatch.setattr(ctrl, "build_plan", lambda intent, question=None: {
        "highlight": [], "sql_legs": [], "doc_leg": None, "api_correlations": []})
    monkeypatch.setattr(ctrl, "_synthesize", lambda *a, **k: "revenue-answer")
    monkeypatch.setattr(ctrl, "check_numeric_grounding", lambda *a, **k: [])

    list(ctrl.answer_stream("What was total revenue"))           # populate
    out = list(ctrl.answer_stream("What is the total revenue"))  # paraphrase -> semantic hit
    assert out[-1]["content"] == "revenue-answer"
    assert out[-1].get("cached") is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && python -m pytest tests/test_query_cache.py::test_answer_stream_serves_semantic_hit -v`
Expected: FAIL — `AttributeError: ... 'embed_query'` / no semantic hit.

- [ ] **Step 3: Implement `embed_query` in `cache.py`**

Add to `backend/semantic_layer/agent/cache.py` (import at top):

```python
from semantic_layer.ingest.llm import get_openai_client
```

```python
def embed_query(question: str) -> list[float]:
    """Embed a question with the configured embedding model for semantic cache lookup."""
    client = get_openai_client()
    resp = client.embeddings.create(
        model=settings.embedding_model,
        input=[question],
        dimensions=settings.embedding_dimensions,
    )
    return resp.data[0].embedding
```

- [ ] **Step 4: Wire semantic lookup into `answer_stream`**

In `backend/semantic_layer/agent/controller.py`, update the import (Task C1 added `query_cache`):

```python
from semantic_layer.agent.cache import query_cache, embed_query
```

Replace the cache-check block from Task C1 (Step 8) with exact-then-semantic, embedding once and reusing it on store:

```python
    q_embedding = None
    if settings.query_cache_enabled:
        cached = query_cache.get_exact(question)
        if cached is None:
            try:
                q_embedding = embed_query(question)
                cached = query_cache.get_semantic(q_embedding, settings.cache_similarity_threshold)
            except Exception:  # noqa: BLE001 — a cache miss must never block answering
                q_embedding = None
        if cached is not None:
            yield {**cached, "cached": True}
            return
```

And update the store call (Task C1 Step 8 end) to persist the embedding:

```python
    if settings.query_cache_enabled:
        query_cache.put(question, answer_event, embedding=q_embedding)
```

- [ ] **Step 5: Run the test + full cache suite**

Run: `cd backend && python -m pytest tests/test_query_cache.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/semantic_layer/agent/cache.py backend/semantic_layer/agent/controller.py backend/tests/test_query_cache.py
git commit -m "feat(cache): semantic (embedding-cosine) query cache layer"
```

---

### Task C3: Document the Redis backend path (production, multi-worker)

**Files:**
- Modify: `docs/superpowers/specs/2026-06-16-neocarta-local-semantic-layer-design.md` (or a new ADR) — documentation only.
- Test: none (doc change).

This task is intentionally **doc-only** (YAGNI): the in-process cache is correct for a single uvicorn worker. Multi-worker / multi-replica deploys need a shared store. Capture the design so it is not lost, but do not build it until a multi-worker deploy is real.

- [ ] **Step 1: Add a "Caching" subsection** to the design spec recording: (a) the in-process `QueryCache` is per-process and does not share across uvicorn workers; (b) the production path is a `RedisQueryCache` implementing the same `get_exact`/`get_semantic`/`put` interface, keyed by normalized question, with semantic lookup backed by a Redis vector index (RediSearch) rather than an in-process linear scan; (c) gate it behind a `cache_backend: str = "memory" | "redis"` setting and a `redis_url`; (d) add `redis>=5` to an optional `pyproject.toml` extra when implemented.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-16-neocarta-local-semantic-layer-design.md
git commit -m "docs(cache): record Redis multi-worker cache backend as follow-up"
```

---

## Self-Review

**Spec coverage** (against the P0 set in the scaling assessment):
- ✅ Schema routing (retrieve→rank, decouple table selection from SQL) — Feature A (A1 retrieval, A2 ranker, A3 wiring behind flag).
- ✅ Batched + parallel entity extraction — Feature B (B1 batch call, B2 parallel runner).
- ✅ Semantic + result caching — Feature C (C1 exact, C2 semantic, C3 Redis follow-up doc).

**Placeholder scan:** No TBDs; every code step shows complete code; every test step shows the assertions; every run step states the expected result.

**Type consistency:**
- `retrieve_candidate_tables -> list[{table_id, score}]` is consumed by `rank_tables(question, candidates, ...)` (reads `c['table_id']`) — consistent (A1↔A2).
- `route_tables -> list[str]` consumed by `build_plan(..., question=...)` which filters `t.startswith("table:sales_pg.sales.")` — consistent (A2↔A3).
- `extract_entities_batch(texts) -> list[list[dict]]` consumed by `extract_entities_for_chunks` via `zip(batch, groups)` — consistent (B1↔B2).
- `QueryCache.get_exact/get_semantic -> dict | None`, `put(question, answer, embedding=None)`, `embed_query -> list[float]` all match controller usage (C1↔C2).
- Controller `build_plan(intent, question=question)` matches the A3 signature `build_plan(intent, question=None)`.

**Out-of-scope (deferred to P1, not in this plan):** weighted k-shortest-path join discovery, async agent + request queue, Postgres connection pooling, incremental/CDC ingestion, observability/tracing, eval rubric, API pagination/auth, MCP tool federation.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-18-production-scaling-p0.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
