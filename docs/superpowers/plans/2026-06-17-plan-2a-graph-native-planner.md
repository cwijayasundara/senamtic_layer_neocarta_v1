# Graph-Native Planner Implementation Plan (Plan 2a of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic, graph-native **planner** — one LLM `extract_intent` pass plus a no-LLM `build_plan` that turns intent into a complete execution blueprint using the knowledge graph (batch value resolution, join paths incl. cross-source bridges, document/period context).

**Architecture:** `extract_intent(question)` is one structured LLM call (powerful `planner_model`) → `Intent`. `build_plan(intent)` is pure Cypher over the graph → `Plan`. This is the core of replacing ~20 LLM discovery round-trips with ≤3 graph queries. The controller + parallel legs + UI rewiring that *consume* the `Plan` are **Plan 2b** (separate).

**Tech Stack:** Python 3.14 + pytest (`cd backend && .venv/bin/python -m pytest …`), Pydantic (already used), LangChain `init_chat_model` / `with_structured_output`, Neo4j Cypher. Live-graph tests are `@pytest.mark.neo4j` + `ingested_graph`; the LLM test is `@pytest.mark.openai`. Docker services must be up.

**Spec:** [Graph-Native Planner + Parallel Legs](../specs/2026-06-17-graph-native-planner-design.md).

**Depends on:** Plan 1 (merged) — `SAME_ENTITY` bridge edges and bridge-aware `get_join_path`.

---

## File Structure

- **Modify** `backend/semantic_layer/config.py` — add `planner_model`, `synthesis_model` + resolved-fallback properties.
- **Modify** `backend/semantic_layer/ingest/llm.py` — `get_chat_model(model=None)`.
- **Create** `backend/semantic_layer/agent/planner.py` — `Intent`, `extract_intent`, `Plan` shape, `build_plan` (+ private Cypher helpers).
- **Create** `backend/tests/test_planner_intent.py` — `extract_intent` (mock unit + 1 live).
- **Create** `backend/tests/test_planner_plan.py` — `build_plan` against the seeded graph.

`Plan` is a plain `dict` (JSON-serializable, so the controller can stream it as an event). `Intent` is a Pydantic model (structured-output target).

---

## Task 1: Per-stage model config

**Files:**
- Modify: `backend/semantic_layer/config.py`
- Test: `backend/tests/test_planner_intent.py` (create with this test first)

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_planner_intent.py
from semantic_layer.config import settings


def test_model_tiering_defaults_fall_back_to_llm_model():
    # Unset planner/synthesis models fall back to the mechanical llm_model.
    assert settings.planner_model_resolved == (settings.planner_model or settings.llm_model)
    assert settings.synthesis_model_resolved == (
        settings.synthesis_model or settings.planner_model or settings.llm_model)
    # With nothing set, both resolve to llm_model.
    if not settings.planner_model and not settings.synthesis_model:
        assert settings.planner_model_resolved == settings.llm_model
        assert settings.synthesis_model_resolved == settings.llm_model
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_planner_intent.py::test_model_tiering_defaults_fall_back_to_llm_model -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'planner_model_resolved'`.

- [ ] **Step 3: Add the settings**

In `backend/semantic_layer/config.py`, add these fields right after `llm_model: str = "openai:gpt-5.4-mini"` (line 26):

```python
    # The powerful model for intent extraction (and synthesis); falls back to
    # llm_model when unset. Set e.g. planner_model=openai:gpt-5.5 in .env to opt in.
    planner_model: str | None = None
    synthesis_model: str | None = None
```

And add these properties to the `Settings` class (next to the existing `postgres_dsn` property):

```python
    @property
    def planner_model_resolved(self) -> str:
        return self.planner_model or self.llm_model

    @property
    def synthesis_model_resolved(self) -> str:
        return self.synthesis_model or self.planner_model or self.llm_model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_planner_intent.py::test_model_tiering_defaults_fall_back_to_llm_model -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/config.py backend/tests/test_planner_intent.py
git commit -m "feat(config): per-stage model tiering (planner_model/synthesis_model)"
```

---

## Task 2: `get_chat_model(model=…)`

**Files:**
- Modify: `backend/semantic_layer/ingest/llm.py`
- Test: `backend/tests/test_planner_intent.py`

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_planner_intent.py`:

```python
def test_get_chat_model_accepts_model_override(monkeypatch):
    import semantic_layer.ingest.llm as llm_mod
    captured = {}
    monkeypatch.setattr(llm_mod, "init_chat_model", lambda m: captured.setdefault("m", m))
    llm_mod.get_chat_model("openai:some-model")
    assert captured["m"] == "openai:some-model"
    llm_mod.get_chat_model()  # default path
    assert captured["m"] == llm_mod.settings.llm_model
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_planner_intent.py::test_get_chat_model_accepts_model_override -v`
Expected: FAIL — `get_chat_model()` takes no arguments → `TypeError`.

- [ ] **Step 3: Add the optional model param**

In `backend/semantic_layer/ingest/llm.py`, change:

```python
def get_chat_model():
    _ensure_key()
    return init_chat_model(settings.llm_model)
```

to:

```python
def get_chat_model(model: str | None = None):
    _ensure_key()
    return init_chat_model(model or settings.llm_model)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_planner_intent.py -v`
Expected: PASS (both tests). Existing callers pass no argument, so they're unaffected.

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/ingest/llm.py backend/tests/test_planner_intent.py
git commit -m "feat(llm): get_chat_model accepts a model override"
```

---

## Task 3: `Intent` + `extract_intent`

**Files:**
- Create: `backend/semantic_layer/agent/planner.py`
- Test: `backend/tests/test_planner_intent.py`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_planner_intent.py`:

```python
import pytest

from semantic_layer.agent.planner import Intent, extract_intent


def test_extract_intent_uses_planner_model_and_returns_intent(monkeypatch):
    import semantic_layer.agent.planner as planner_mod
    canned = Intent(terms=["EMEA", "Blackwell"], needs_sql=True, needs_doc=True)
    captured = {}

    class _FakeStructured:
        def invoke(self, _messages):
            return canned

    class _FakeModel:
        def with_structured_output(self, schema):
            captured["schema"] = schema
            return _FakeStructured()

    monkeypatch.setattr(planner_mod, "get_chat_model",
                        lambda model=None: captured.setdefault("model", model) or _FakeModel())
    out = extract_intent("which EMEA Blackwell things, per the press release?")
    assert out is canned
    assert captured["schema"] is Intent
    assert captured["model"] == planner_mod.settings.planner_model_resolved


@pytest.mark.openai
def test_extract_intent_live_pulls_terms(require_openai):
    out = extract_intent(
        "In FY2025, which EMEA Cloud customers bought Blackwell Data Center products?")
    lowered = [t.lower() for t in out.terms]
    assert any("emea" in t for t in lowered)
    assert any("blackwell" in t for t in lowered)
    assert out.needs_sql is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/python -m pytest tests/test_planner_intent.py -k extract_intent -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'semantic_layer.agent.planner'`.

- [ ] **Step 3: Create the planner module with `Intent` + `extract_intent`**

Create `backend/semantic_layer/agent/planner.py`:

```python
"""Graph-native planner: one LLM intent pass, then deterministic graph planning.

extract_intent(question) -> Intent     (one structured LLM call, planner_model)
build_plan(intent)       -> Plan dict   (pure Cypher; added in a later task)

This replaces the orchestrator's ~20 LLM discovery round-trips with a single intent
read plus a few set-based graph queries.
"""

from pydantic import BaseModel, Field

from semantic_layer.config import settings
from semantic_layer.ingest.llm import get_chat_model

_INTENT_PROMPT = (
    "You read a business question over an NVIDIA enterprise semantic layer that unifies "
    "SQL databases, REST APIs, and documents. Extract a structured intent.\n"
    "- terms: the dimension FILTER descriptors mentioned (e.g. 'EMEA','Cloud','Blackwell',"
    "'Data Center'). Split compound noun phrases into separate descriptors.\n"
    "- fact: the measure/metric in plain words (e.g. 'revenue','gpu usage','open tickets'), or null.\n"
    "- group_by: dimensions to break results down by (e.g. ['customer','quarter']).\n"
    "- fiscal_year / quarter: a fiscal scope if stated (e.g. 2025 / 'Q1'), else null.\n"
    "- needs_sql / needs_api / needs_doc: which source TYPES the question requires.\n"
    "- doc_query: what to look up in the documents, or null.\n"
    "- api_intents: enterprise-system lookups implied (e.g. ['dgx usage','open tickets'])."
)


class Intent(BaseModel):
    terms: list[str] = Field(default_factory=list)
    fact: str | None = None
    group_by: list[str] = Field(default_factory=list)
    fiscal_year: int | None = None
    quarter: str | None = None
    needs_sql: bool = True
    needs_api: bool = False
    needs_doc: bool = False
    doc_query: str | None = None
    api_intents: list[str] = Field(default_factory=list)


def extract_intent(question: str) -> Intent:
    """One structured LLM call (planner_model) → Intent."""
    model = get_chat_model(settings.planner_model_resolved).with_structured_output(Intent)
    return model.invoke([("system", _INTENT_PROMPT), ("human", question)])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_planner_intent.py -v`
Expected: PASS — the mock test passes; the live `@pytest.mark.openai` test passes if the OpenAI key is set (else it skips).

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/agent/planner.py backend/tests/test_planner_intent.py
git commit -m "feat(agent): Intent model + extract_intent (structured LLM, planner_model)"
```

---

## Task 4: `build_plan` — value resolution + SQL join targets

**Files:**
- Modify: `backend/semantic_layer/agent/planner.py`
- Test: `backend/tests/test_planner_plan.py`

Background (verified seed): `norm()` keys match `:Value` nodes — 'EMEA'→`sales.region.name`, 'Cloud'→`sales.industry.name`='Cloud Service Provider', 'Blackwell'→`sales.architecture.name`, 'Data Center'→`sales.segment.name`. The sales fact table is `table:sales_pg.sales.order_line`. `get_join_path` (bridge-aware, from Plan 1) returns `{found, tables, joins}`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_planner_plan.py
import pytest

from semantic_layer.agent.planner import Intent, build_plan


@pytest.mark.neo4j
def test_build_plan_resolves_values_and_join_targets(ingested_graph):
    intent = Intent(terms=["EMEA", "Cloud", "Blackwell", "Data Center"],
                    fact="revenue", needs_sql=True)
    plan = build_plan(intent)

    resolved = {r["term"]: r for r in plan["resolved_values"]}
    # All four descriptors resolve to a sales_pg dimension column with exact spelling.
    assert resolved["Cloud"]["source"] == "sales_pg"
    assert resolved["Cloud"]["exact"] == "Cloud Service Provider"
    assert resolved["Blackwell"]["table_id"] == "table:sales_pg.sales.architecture"

    assert len(plan["sql_legs"]) >= 1
    sales = next(leg for leg in plan["sql_legs"] if leg["source"] == "sales_pg")
    assert sales["fact_table"] == "table:sales_pg.sales.order_line"
    # Every resolved sales dimension is a join target with a concrete join chain.
    target_tables = {jt["table_id"] for jt in sales["join_targets"]}
    assert "table:sales_pg.sales.segment" in target_tables
    assert all(jt["joins"] for jt in sales["join_targets"])
    # Filters carry the EXACT stored spelling for the SQL leg to apply.
    filt = {f["column"]: f["value"] for f in sales["filters"]}
    assert filt.get("name") is not None  # dimension name columns filtered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_planner_plan.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_plan'`.

- [ ] **Step 3: Implement value resolution + join targets**

Append to `backend/semantic_layer/agent/planner.py`:

```python
import json

from semantic_layer.agent.driver import driver
from semantic_layer.agent.graph_tools import get_join_path
from semantic_layer.ingest.value_indexer import norm

_SALES_FACT = "table:sales_pg.sales.order_line"

_RESOLVE_CYPHER = """
UNWIND $rows AS row
MATCH (db:Database)-[:HAS_SCHEMA]->(:Schema)-[:HAS_TABLE]->(t:Table)
     -[:HAS_COLUMN]->(c:Column)-[:HAS_VALUE]->(v:Value {norm: row.norm})
RETURN row.term AS term, db.name AS source, t.id AS table_id,
       c.name AS column, v.name AS exact
"""


def _resolve_values(terms: list[str]) -> list[dict]:
    rows = [{"term": t, "norm": norm(t)} for t in terms]
    recs = driver().execute_query(
        _RESOLVE_CYPHER, rows=rows, database_=settings.neo4j_database,
    ).records
    return [dict(r) for r in recs]


def _join_targets(fact: str, table_ids: list[str]) -> list[dict]:
    out = []
    for tid in dict.fromkeys(table_ids):  # de-dupe, preserve order
        path = json.loads(get_join_path.invoke({"table_a_id": fact, "table_b_id": tid}))
        if path.get("found"):
            out.append({"table_id": tid, "tables": path["tables"], "joins": path["joins"]})
    return out


def build_plan(intent: "Intent") -> dict:
    """Deterministic graph planning. No LLM. Returns a JSON-serializable Plan dict."""
    resolved = _resolve_values(intent.terms)

    sql_legs = []
    sales_dims = [r for r in resolved if r["source"] == "sales_pg"]
    if sales_dims:
        sql_legs.append({
            "source": "sales_pg",
            "fact_table": _SALES_FACT,
            "join_targets": _join_targets(_SALES_FACT, [r["table_id"] for r in sales_dims]),
            "filters": [{"table_id": r["table_id"], "column": r["column"], "value": r["exact"]}
                        for r in sales_dims],
        })

    return {"resolved_values": resolved, "sql_legs": sql_legs}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_planner_plan.py -v`
Expected: PASS. (If the local graph predates Plan 1's bridge edges, run `cd backend && .venv/bin/python -m semantic_layer.ingest.pipeline` once first.)

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/agent/planner.py backend/tests/test_planner_plan.py
git commit -m "feat(agent): build_plan value resolution + SQL join targets"
```

---

## Task 5: `build_plan` — document/period context + API correlation + highlight

**Files:**
- Modify: `backend/semantic_layer/agent/planner.py`
- Test: `backend/tests/test_planner_plan.py`

Background (verified): document entities bridge to values (`Entity-[:REFERS_TO]->Value`), chunks mention entities (`Chunk-[:MENTIONS]->Entity`), documents own chunks (`Document-[:HAS_CHUNK]->Chunk`) and cover periods (`Document-[:COVERS_PERIOD]->Period`). API endpoints bridge to SQL via `SAME_ENTITY` (Plan 1).

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_planner_plan.py`:

```python
@pytest.mark.neo4j
def test_build_plan_adds_doc_context_api_keys_and_highlight(ingested_graph):
    intent = Intent(terms=["Blackwell", "Data Center"], fact="revenue",
                    needs_sql=True, needs_doc=True, needs_api=True,
                    doc_query="what drove Data Center growth",
                    api_intents=["dgx usage", "open tickets"])
    plan = build_plan(intent)

    # Documents that mention the entities are surfaced as candidates.
    assert plan["doc_leg"] is not None
    assert "doc:NVIDIAAn_2026" in plan["doc_leg"]["candidate_doc_ids"]
    assert plan["doc_leg"]["doc_query"] == "what drove Data Center growth"

    # API correlation keys come straight from the SAME_ENTITY bridge.
    pairs = {(k["sql_column"], k["api_column"]) for k in plan["api_correlations"]}
    assert ("col:sales_pg.sales.customer.customer_id",
            "col:itsm.api.GET /tickets.account_id") in pairs

    # Highlight is the union of plan node ids for the UI graph.
    assert "table:sales_pg.sales.segment" in plan["highlight"]
    assert "doc:NVIDIAAn_2026" in plan["highlight"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_planner_plan.py::test_build_plan_adds_doc_context_api_keys_and_highlight -v`
Expected: FAIL — `KeyError: 'doc_leg'` (not yet produced).

- [ ] **Step 3: Implement context docs, API correlations, and highlight**

In `backend/semantic_layer/agent/planner.py`, add these Cypher helpers and extend `build_plan`. Add the helpers after `_join_targets`:

```python
_DOC_CYPHER = """
UNWIND $norms AS n
MATCH (e:Entity)-[:REFERS_TO]->(:Value {norm: n})
MATCH (ch:Chunk)-[:MENTIONS]->(e), (d:Document)-[:HAS_CHUNK]->(ch)
OPTIONAL MATCH (d)-[:COVERS_PERIOD]->(p:Period)
RETURN DISTINCT d.id AS doc_id, collect(DISTINCT p.key) AS periods
"""

_API_CORR_CYPHER = """
MATCH (api:Column)-[:SAME_ENTITY]->(sql:Column)
RETURN sql.id AS sql_column, api.id AS api_column ORDER BY api_column
"""


def _context_docs(terms: list[str]) -> dict | None:
    norms = [norm(t) for t in terms]
    recs = driver().execute_query(
        _DOC_CYPHER, norms=norms, database_=settings.neo4j_database,
    ).records
    doc_ids = [r["doc_id"] for r in recs]
    periods = sorted({p for r in recs for p in r["periods"] if p})
    if not doc_ids:
        return None
    return {"candidate_doc_ids": doc_ids, "periods": periods}


def _api_correlations() -> list[dict]:
    recs = driver().execute_query(
        _API_CORR_CYPHER, database_=settings.neo4j_database,
    ).records
    return [{"sql_column": r["sql_column"], "api_column": r["api_column"]} for r in recs]
```

Then replace the `return {...}` at the end of `build_plan` with:

```python
    doc_leg = None
    if intent.needs_doc:
        ctx = _context_docs(intent.terms)
        if ctx is not None:
            doc_leg = {"doc_query": intent.doc_query, **ctx}

    api_correlations = _api_correlations() if intent.needs_api else []

    highlight = sorted({
        *(r["table_id"] for r in resolved),
        *(t for leg in sql_legs for jt in leg["join_targets"] for t in jt["tables"]),
        *(doc_leg["candidate_doc_ids"] if doc_leg else []),
    })

    return {
        "resolved_values": resolved,
        "sql_legs": sql_legs,
        "doc_leg": doc_leg,
        "api_correlations": api_correlations,
        "highlight": highlight,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_planner_plan.py -v`
Expected: PASS — both `build_plan` tests. (Re-ingest once first if the local graph predates Plan 1.)

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/agent/planner.py backend/tests/test_planner_plan.py
git commit -m "feat(agent): build_plan doc/period context, API correlations, highlight"
```

---

## Self-Review notes

- **Spec coverage:** `extract_intent` (Task 3) ✓; `build_plan` batch value resolution (Task 4) ✓; join paths incl. bridges via `get_join_path` (Task 4) ✓; context-graph documents+periods (Task 5) ✓; API correlation keys from `SAME_ENTITY` (Task 5) ✓; per-stage model tiering (Tasks 1–2) ✓; highlight set for the UI (Task 5) ✓.
- **Deferred to Plan 2b (controller):** `answer_stream` + the 4-stage drive, parallel `ThreadPoolExecutor` legs, `run_sql_leg`/`run_api_leg`/`run_doc_leg`, `web/events.py` rewiring, reverting the recursion-cap band-aid, synthesis with `synthesis_model`, and the end-to-end bounded-LLM-call test. These consume the `Plan`/`Intent` types defined here.
- **Leg independence / API correlation:** per the spec, API legs are not SQL-joined across the bridge; `build_plan` exposes `api_correlations` (the `customer_id ↔ account_id` pairs) so the controller can correlate at synthesis time. This is why Task 4's join-target drop of the final bridge column (noted in Plan 1) is harmless here.
- **`Plan` is a dict** (JSON-serializable) so Plan 2b's controller can stream it as a `tool_result` event for the trace/graph UI.
- **Type consistency:** `Intent` fields (`terms`, `needs_sql/api/doc`, `doc_query`, `api_intents`, `fiscal_year`, `quarter`) are used identically in `extract_intent` and `build_plan`. `Plan` keys (`resolved_values`, `sql_legs[{source,fact_table,join_targets,filters}]`, `doc_leg`, `api_correlations`, `highlight`) are consistent across Tasks 4–5.
- **Reminder:** running backend pytest can wipe doc-chunk embeddings; the doc-context test in Task 5 needs entities/values present — if it returns no docs, run `make ingest` (or `python -m semantic_layer.ingest.pipeline`) and retry.
```
