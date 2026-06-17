# Controller + Parallel Legs Implementation Plan (Plan 2b of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the graph-native planner (Plan 2a) into the live request path: a deterministic controller that runs `extract_intent → build_plan → parallel legs → synthesize`, replacing the free-form orchestrator on the web path with a bounded, parallel, graph-planned flow.

**Architecture:** Each leg (`run_sql_leg`/`run_api_leg`/`run_doc_leg`) is a focused worker that executes a pre-resolved plan slice with **one structured LLM call** (`llm_model`) plus a deterministic tool call — no discovery. `controller.answer_stream` drives the stages, fans the legs out concurrently with a `ThreadPoolExecutor`, emits the existing UI event stream, and synthesizes a grounded answer with `synthesis_model`. Bounded LLM calls ≈ extract(1) + legs(≤1 each) + synth(1).

**Tech Stack:** Python 3.14 + pytest, Pydantic structured output (`with_structured_output`), `concurrent.futures.ThreadPoolExecutor`, Neo4j/Postgres/SQLite, the in-process API `TestClient`. Unit tests mock the LLM (no key needed); live tests are `@pytest.mark.openai`/`@pytest.mark.neo4j`.

**Spec:** [Graph-Native Planner + Parallel Legs](../specs/2026-06-17-graph-native-planner-design.md).

**Depends on:** Plan 1 (bridges) + Plan 2a (`Intent`, `extract_intent`, `build_plan`) — both merged.

---

## File Structure

- **Modify** `backend/semantic_layer/agent/planner.py` — extend `Intent` (`financial_metrics`), add a financials direct leg + `scope` to `build_plan`.
- **Create** `backend/semantic_layer/agent/legs.py` — `run_sql_leg`, `run_api_leg`, `run_doc_leg` (+ private helpers).
- **Create** `backend/semantic_layer/agent/controller.py` — `answer_stream(question)`.
- **Modify** `backend/semantic_layer/web/events.py` — delegate `stream_chat_events` to the controller; drop the recursion-cap band-aid.
- **Tests:** `backend/tests/test_legs.py`, `backend/tests/test_controller.py`, and updates to `backend/tests/test_planner_plan.py` + `backend/tests/test_web_chat_events.py`.

Leg result shapes (consumed by the controller):
```python
# run_sql_leg -> matches the answer event's sql_runs[] entry
{"source", "sql", "columns": [...], "rows": [...], "row_count": int, "error": str | None}
# run_api_leg
{"calls": [{"source","path","params","status","row_count","data"}], "error": str | None}
# run_doc_leg
{"answer": str, "citations": [{"doc_id","chunk_id","quote","score"}], "doc_texts": [str], "error": str | None}
```

---

## Task 1: Plan a financials leg + uniform `scope`

**Files:**
- Modify: `backend/semantic_layer/agent/planner.py`
- Test: `backend/tests/test_planner_plan.py`

`build_plan` currently emits only a sales `sql_leg`. Company-financials questions ("revenue and gross margin by quarter") need a direct `income_statement` leg. Add an explicit `Intent.financial_metrics` (LLM-populated) rather than guessing from `fact`.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_planner_plan.py`:

```python
@pytest.mark.neo4j
def test_build_plan_adds_financials_leg_and_scope(ingested_graph):
    intent = Intent(terms=["Blackwell"], needs_sql=True,
                    financial_metrics=["revenue", "gross margin"],
                    fiscal_year=2027, quarter="Q1")
    plan = build_plan(intent)
    fin = next((leg for leg in plan["sql_legs"] if leg["source"] == "financials"), None)
    assert fin is not None
    assert fin["fact_table"] == "table:financials.main.income_statement"
    assert fin["join_targets"] == [] and fin["filters"] == []
    assert fin["scope"] == {"fiscal_year": 2027, "quarter": "Q1"}
    # every sql_leg now carries a scope (the sales leg too)
    assert all("scope" in leg for leg in plan["sql_legs"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_planner_plan.py::test_build_plan_adds_financials_leg_and_scope -v`
Expected: FAIL — `Intent` has no `financial_metrics` (TypeError) — or, once that's added, no financials leg.

- [ ] **Step 3: Implement**

In `backend/semantic_layer/agent/planner.py`, add a field to `Intent` (after `api_intents`):

```python
    financial_metrics: list[str] = Field(default_factory=list)
```

Add this line to the `_INTENT_PROMPT` (before the closing `)`), so the LLM populates it:

```python
    "\n- financial_metrics: company-wide financial measures asked for "
    "(e.g. ['revenue','gross margin','net income']); empty unless the question asks "
    "about overall company financials."
```

In `build_plan`, compute a `scope` and append the financials leg. Replace the `sql_legs = []` / sales-leg block with:

```python
    scope = {"fiscal_year": intent.fiscal_year, "quarter": intent.quarter}
    sql_legs = []
    sales_dims = [r for r in resolved if r["source"] == "sales_pg"]
    if sales_dims:
        sql_legs.append({
            "source": "sales_pg",
            "fact_table": _SALES_FACT,
            "join_targets": _join_targets(_SALES_FACT, [r["table_id"] for r in sales_dims]),
            "filters": [{"table_id": r["table_id"], "column": r["column"], "value": r["exact"]}
                        for r in sales_dims],
            "scope": scope,
        })
    if intent.financial_metrics:
        sql_legs.append({
            "source": "financials",
            "fact_table": "table:financials.main.income_statement",
            "join_targets": [], "filters": [], "scope": scope,
            "metrics": intent.financial_metrics,
        })
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_planner_plan.py -v`
Expected: PASS (the new test + the two existing `build_plan` tests — they don't assert exact leg dicts, so the added `scope` key is compatible).

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/agent/planner.py backend/tests/test_planner_plan.py
git commit -m "feat(agent): plan a financials direct leg + uniform scope"
```

---

## Task 2: `run_sql_leg`

**Files:**
- Create: `backend/semantic_layer/agent/legs.py`
- Test: `backend/tests/test_legs.py`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_legs.py
import json
import pytest

from semantic_layer.agent import legs as legs_mod
from semantic_layer.agent.legs import run_sql_leg


class _FakeStructured:
    def __init__(self, value):
        self._value = value

    def invoke(self, _messages):
        return self._value


class _FakeModel:
    def __init__(self, draft):
        self._draft = draft

    def with_structured_output(self, _schema):
        return _FakeStructured(self._draft)


def test_run_sql_leg_executes_drafted_sql(monkeypatch):
    leg = {"source": "financials", "fact_table": "table:financials.main.income_statement",
           "join_targets": [], "filters": [], "scope": {"fiscal_year": 2027, "quarter": "Q1"},
           "metrics": ["revenue"]}
    draft = legs_mod._SqlDraft(sql="SELECT revenue FROM income_statement LIMIT 1")
    monkeypatch.setattr(legs_mod, "get_chat_model", lambda model=None: _FakeModel(draft))
    captured = {}
    monkeypatch.setattr(legs_mod, "_run",
                        lambda source, sql: captured.update(source=source, sql=sql)
                        or json.dumps({"columns": ["revenue"], "rows": [[67795.0]]}))
    out = run_sql_leg(leg)
    assert captured["source"] == "financials"
    assert out["sql"].lower().startswith("select")
    assert out["columns"] == ["revenue"]
    assert out["row_count"] == 1
    assert out["error"] is None


def test_run_sql_leg_retries_once_on_error(monkeypatch):
    leg = {"source": "sales_pg", "fact_table": "table:sales_pg.sales.order_line",
           "join_targets": [], "filters": [], "scope": {"fiscal_year": None, "quarter": None}}
    draft = legs_mod._SqlDraft(sql="SELECT bad")
    monkeypatch.setattr(legs_mod, "get_chat_model", lambda model=None: _FakeModel(draft))
    calls = []
    monkeypatch.setattr(legs_mod, "_run",
                        lambda source, sql: calls.append(sql)
                        or json.dumps({"error": "boom"}))
    out = run_sql_leg(leg)
    assert len(calls) == 2          # initial + one retry
    assert out["error"] == "boom"
    assert out["rows"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/python -m pytest tests/test_legs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'semantic_layer.agent.legs'`.

- [ ] **Step 3: Create `backend/semantic_layer/agent/legs.py`**

```python
"""Leg workers: execute one pre-resolved plan slice with a single structured LLM call.

Each leg receives an exact slice of the Plan (tables, joins, filters, endpoints) — no
discovery — writes/selects what to run, executes a deterministic tool, and returns a
structured result the controller folds into the answer.
"""

import json

from pydantic import BaseModel

from semantic_layer.agent.graph_tools import _sql_reference
from semantic_layer.agent.sql_tools import _run
from semantic_layer.ingest.llm import get_chat_model
from semantic_layer.config import settings


class _SqlDraft(BaseModel):
    sql: str


_SQL_LEG_PROMPT = (
    "You are a SQL expert. Write ONE read-only SELECT for the given plan slice and nothing "
    "else. Use the exact table references and join column pairs provided. Match dimension "
    "name filters case-insensitively with ILIKE '%value%'. If a fiscal scope is given and "
    "the fact is order_line, reach the period via order_line->sales_order(order_id)->"
    "fiscal_period(fiscal_period_id) and filter fiscal_year/quarter; if the fact is "
    "income_statement, filter its own fiscal_year/quarter columns. Return only the SQL."
)


def _col(cid: str) -> str:
    """Readable table.column from a column id col:src.schema.table.column."""
    parts = cid.split(":", 1)[1].split(".")
    return f"{parts[-2]}.{parts[-1]}"


def _sql_brief(leg: dict) -> str:
    lines = [f"Source: {leg['source']}",
             f"Fact table: {_sql_reference(leg['fact_table'])}"]
    for jt in leg.get("join_targets", []):
        pairs = ", ".join(f"{_col(j['on'][0])} = {_col(j['on'][1])}" for j in jt["joins"])
        lines.append(f"Join {_sql_reference(jt['table_id'])} ON {pairs}")
    for f in leg.get("filters", []):
        lines.append(f"Filter {_sql_reference(f['table_id'])}.{f['column']} ~ '{f['value']}'")
    scope = leg.get("scope") or {}
    if scope.get("fiscal_year"):
        lines.append(f"Scope: fiscal_year={scope['fiscal_year']} quarter={scope.get('quarter')}")
    if leg.get("metrics"):
        lines.append(f"Select these measures: {', '.join(leg['metrics'])}")
    return "\n".join(lines)


def _draft_and_run(model, brief: str, source: str, extra: str = "") -> tuple[str, dict]:
    draft = model.invoke([("system", _SQL_LEG_PROMPT), ("human", brief + extra)])
    return draft.sql, json.loads(_run(source, draft.sql))


def run_sql_leg(leg: dict) -> dict:
    brief = _sql_brief(leg)
    model = get_chat_model(settings.llm_model).with_structured_output(_SqlDraft)
    sql, out = _draft_and_run(model, brief, leg["source"])
    if isinstance(out, dict) and "error" in out:
        sql, out = _draft_and_run(model, brief, leg["source"],
                                  extra=f"\n\nThe previous SQL failed: {out['error']}\nFix it.")
    if isinstance(out, dict) and "error" in out:
        return {"source": leg["source"], "sql": sql, "columns": [], "rows": [],
                "row_count": 0, "error": out["error"]}
    rows = out.get("rows", [])
    return {"source": leg["source"], "sql": sql, "columns": out.get("columns", []),
            "rows": rows, "row_count": len(rows), "error": None}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_legs.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/agent/legs.py backend/tests/test_legs.py
git commit -m "feat(agent): run_sql_leg (structured SQL from plan slice + retry)"
```

---

## Task 3: `run_api_leg`

**Files:**
- Modify: `backend/semantic_layer/agent/legs.py`
- Test: `backend/tests/test_legs.py`

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_legs.py`:

```python
from semantic_layer.agent.legs import run_api_leg


def test_run_api_leg_executes_planned_calls(monkeypatch):
    plan_calls = legs_mod._ApiCalls(calls=[
        legs_mod._ApiCall(source="itsm", path="/tickets", params={"status": "open"})])
    monkeypatch.setattr(legs_mod, "get_chat_model", lambda model=None: _FakeModel(plan_calls))
    monkeypatch.setattr(legs_mod, "call_api",
                        lambda source, path, params=None: json.dumps(
                            {"status": 200, "data": [{"id": 1}, {"id": 2}]}))
    out = run_api_leg(["open tickets"])
    assert out["error"] is None
    assert out["calls"][0]["source"] == "itsm"
    assert out["calls"][0]["path"] == "/tickets"
    assert out["calls"][0]["row_count"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_legs.py::test_run_api_leg_executes_planned_calls -v`
Expected: FAIL — `cannot import name 'run_api_leg'`.

- [ ] **Step 3: Implement** — append to `backend/semantic_layer/agent/legs.py`:

```python
from semantic_layer.agent.api_tools import call_api


class _ApiCall(BaseModel):
    source: str
    path: str
    params: dict = {}


class _ApiCalls(BaseModel):
    calls: list[_ApiCall] = []


_API_LEG_PROMPT = (
    "Pick the mock-API calls that answer the given lookups. Sources & endpoints: "
    "crm (/accounts,/contacts,/opportunities), itsm (/tickets,/rma), "
    "partner (/partners,/inventory), dgx (/usage). Use query params to filter "
    "(e.g. {'status':'open'} for open tickets). Return the list of calls to make."
)


def run_api_leg(api_intents: list[str]) -> dict:
    model = get_chat_model(settings.llm_model).with_structured_output(_ApiCalls)
    plan = model.invoke([("system", _API_LEG_PROMPT),
                         ("human", "Lookups: " + "; ".join(api_intents))])
    results = []
    for c in plan.calls:
        resp = json.loads(call_api(c.source, c.path, c.params))
        body = resp.get("data")
        row_count = len(body) if isinstance(body, list) else (1 if body else 0)
        results.append({"source": c.source, "path": c.path, "params": c.params,
                        "status": resp.get("status"), "row_count": row_count, "data": body})
    return {"calls": results, "error": None}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_legs.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/agent/legs.py backend/tests/test_legs.py
git commit -m "feat(agent): run_api_leg (structured endpoint/params + execute)"
```

---

## Task 4: `run_doc_leg`

**Files:**
- Modify: `backend/semantic_layer/agent/legs.py`
- Test: `backend/tests/test_legs.py`

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_legs.py`:

```python
from semantic_layer.agent.legs import run_doc_leg


def test_run_doc_leg_retrieves_and_answers(monkeypatch):
    monkeypatch.setattr(legs_mod, "search_documents", type("T", (), {
        "invoke": staticmethod(lambda args: json.dumps([
            {"chunk_id": "doc:x:chunk:2", "doc_id": "doc:x",
             "text": "Data Center revenue was a record $60.4 billion.", "score": 0.9}]))})())
    answer = legs_mod._DocAnswer(answer="Data Center revenue was $60.4 billion (doc:x).")
    monkeypatch.setattr(legs_mod, "get_chat_model", lambda model=None: _FakeModel(answer))
    out = run_doc_leg("what drove Data Center growth")
    assert out["error"] is None
    assert out["citations"][0]["doc_id"] == "doc:x"
    assert "60.4" in out["answer"]
    assert any("60.4" in t for t in out["doc_texts"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_legs.py::test_run_doc_leg_retrieves_and_answers -v`
Expected: FAIL — `cannot import name 'run_doc_leg'`.

- [ ] **Step 3: Implement** — append to `backend/semantic_layer/agent/legs.py`:

```python
from semantic_layer.agent.doc_tools import search_documents


class _DocAnswer(BaseModel):
    answer: str


_DOC_LEG_PROMPT = (
    "Answer the question ONLY from the provided document passages, quoting the most "
    "relevant sentence and citing the document id. If nothing relevant is present, say so."
)


def run_doc_leg(doc_query: str) -> dict:
    hits = json.loads(search_documents.invoke({"query": doc_query}))
    citations, doc_texts, seen = [], [], set()
    for h in hits:
        cid = h.get("chunk_id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        text = h.get("text") or ""
        doc_texts.append(text)
        citations.append({"doc_id": h.get("doc_id"), "chunk_id": cid,
                          "quote": text[:280], "score": h.get("score")})
    passages = "\n\n".join(f"[{c['doc_id']}] {t}" for c, t in zip(citations, doc_texts))
    model = get_chat_model(settings.llm_model).with_structured_output(_DocAnswer)
    ans = model.invoke([("system", _DOC_LEG_PROMPT),
                        ("human", f"Question: {doc_query}\n\nPassages:\n{passages}")])
    return {"answer": ans.answer, "citations": citations, "doc_texts": doc_texts, "error": None}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_legs.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/agent/legs.py backend/tests/test_legs.py
git commit -m "feat(agent): run_doc_leg (vector retrieve + grounded quote)"
```

---

## Task 5: `controller.answer_stream`

**Files:**
- Create: `backend/semantic_layer/agent/controller.py`
- Test: `backend/tests/test_controller.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_controller.py
from semantic_layer.agent import controller as ctrl
from semantic_layer.agent.planner import Intent


def _fake_synth(*_a, **_k):
    return "Segment Data Center leads; per doc:x revenue was $60.4 billion."


def test_answer_stream_runs_stages_and_emits_answer(monkeypatch):
    monkeypatch.setattr(ctrl, "extract_intent",
                        lambda q: Intent(terms=["Data Center"], needs_sql=True, needs_doc=True,
                                         doc_query="dc growth"))
    monkeypatch.setattr(ctrl, "build_plan", lambda intent: {
        "resolved_values": [], "highlight": ["table:sales_pg.sales.segment", "doc:x"],
        "sql_legs": [{"source": "sales_pg", "fact_table": "table:sales_pg.sales.order_line",
                      "join_targets": [], "filters": [], "scope": {}}],
        "doc_leg": {"doc_query": "dc growth", "candidate_doc_ids": ["doc:x"], "periods": []},
        "api_correlations": [],
    })
    monkeypatch.setattr(ctrl, "run_sql_leg", lambda leg: {
        "source": "sales_pg", "sql": "SELECT 1", "columns": ["n"], "rows": [[60400000000]],
        "row_count": 1, "error": None})
    monkeypatch.setattr(ctrl, "run_doc_leg", lambda q: {
        "answer": "doc says $60.4 billion", "citations": [
            {"doc_id": "doc:x", "chunk_id": "doc:x:chunk:2", "quote": "$60.4 billion", "score": 0.9}],
        "doc_texts": ["Data Center revenue was a record $60.4 billion."], "error": None})
    monkeypatch.setattr(ctrl, "_synthesize", _fake_synth)

    events = list(ctrl.answer_stream("what drove Data Center growth?"))
    answer = events[-1]
    assert answer["type"] == "answer"
    assert answer["sql_runs"][0]["row_count"] == 1
    assert answer["doc_citations"][0]["doc_id"] == "doc:x"
    assert "table:sales_pg.sales.segment" in answer["highlight"]
    assert answer["caveats"] == []                       # $60.4B grounded by doc_texts
    # bounded: exactly one plan_query event + one event per leg + the answer
    kinds = [e["type"] for e in events]
    assert kinds.count("answer") == 1
    assert "tool_result" in kinds
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_controller.py -v`
Expected: FAIL — `No module named 'semantic_layer.agent.controller'`.

- [ ] **Step 3: Create `backend/semantic_layer/agent/controller.py`**

```python
"""Deterministic controller: extract_intent -> build_plan -> parallel legs -> synthesize.

Emits the same UI event stream as the legacy path (tool_call/tool_result/answer) so the
web UI is unchanged. Bounded LLM calls: extract(1) + legs(<=1 each) + synthesize(1).
"""

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

from semantic_layer.agent.planner import extract_intent, build_plan
from semantic_layer.agent.legs import run_sql_leg, run_api_leg, run_doc_leg
from semantic_layer.ingest.llm import get_chat_model
from semantic_layer.config import settings
from semantic_layer.web.grounding import check_numeric_grounding

_SYNTH_PROMPT = (
    "Synthesize a concise answer from the leg results below. State which source(s) each "
    "fact came from and cite document ids. Only use numbers present in the results. "
    "If the API results need to be tied to specific customers, correlate via the given keys."
)


def _synthesize(question: str, sql_runs, api_calls, doc, correlations) -> str:
    payload = {"question": question, "sql": sql_runs, "api": api_calls,
               "doc_answer": (doc or {}).get("answer"), "api_correlations": correlations}
    model = get_chat_model(settings.synthesis_model_resolved)
    msg = model.invoke([("system", _SYNTH_PROMPT), ("human", json.dumps(payload, default=str))])
    return msg.content


def answer_stream(question: str) -> Iterator[dict]:
    try:
        intent = extract_intent(question)
        plan = build_plan(intent)
        yield {"type": "tool_result", "scope": "plan", "name": "plan_query",
               "content": json.dumps({k: plan[k] for k in ("highlight",) if k in plan})[:4000]}

        # Fan out independent legs concurrently.
        jobs = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            for leg in plan.get("sql_legs", []):
                jobs[pool.submit(run_sql_leg, leg)] = ("sql", leg["source"])
            if intent.needs_api and intent.api_intents:
                jobs[pool.submit(run_api_leg, intent.api_intents)] = ("api", "api")
            if plan.get("doc_leg"):
                jobs[pool.submit(run_doc_leg, plan["doc_leg"]["doc_query"])] = ("doc", "doc")

            sql_runs, api_calls, doc_texts, doc_citations, doc = [], [], [], [], None
            for fut in list(jobs):
                kind, label = jobs[fut]
                res = fut.result()
                yield {"type": "tool_result", "scope": kind, "name": f"{kind}_leg",
                       "content": json.dumps(res, default=str)[:4000]}
                if kind == "sql":
                    sql_runs.append(res)
                elif kind == "api":
                    api_calls.extend(res.get("calls", []))
                elif kind == "doc":
                    doc = res
                    doc_citations = res.get("citations", [])
                    doc_texts = res.get("doc_texts", [])

        summary = _synthesize(question, sql_runs, api_calls, doc,
                              plan.get("api_correlations", []))
        caveats = check_numeric_grounding(summary, sql_runs, api_calls, doc_texts)
    except Exception as exc:  # noqa: BLE001 — never leave the UI hanging
        yield {"type": "answer", "content": f"The agent could not complete this question: {exc}",
               "highlight": [], "sql_runs": [], "api_calls": [], "doc_citations": [], "caveats": []}
        return

    yield {"type": "answer", "content": summary, "highlight": plan.get("highlight", []),
           "sql_runs": sql_runs, "api_calls": api_calls,
           "doc_citations": doc_citations, "caveats": caveats}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_controller.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/agent/controller.py backend/tests/test_controller.py
git commit -m "feat(agent): answer_stream controller (parallel legs + grounded synthesis)"
```

---

## Task 6: Wire the web path to the controller

**Files:**
- Modify: `backend/semantic_layer/web/events.py`
- Test: `backend/tests/test_web_chat_events.py`

- [ ] **Step 1: Update the end-to-end test**

In `backend/tests/test_web_chat_events.py`, the existing `test_answer_event_carries_sql_provenance` should still pass via the new path (it asserts structured fields). Append a controller-path assertion test:

```python
@pytest.mark.neo4j
@pytest.mark.postgres
@pytest.mark.openai
def test_chat_uses_controller_and_is_bounded(ingested_graph, require_openai):
    events = list(stream_chat_events(
        "Which segment has the most revenue? Use the sales database."))
    answer = events[-1]
    assert answer["type"] == "answer"
    assert answer["sql_runs"], "controller should surface the SQL leg"
    # No plan should require the old discovery loop: there is a plan_query event.
    assert any(e.get("name") == "plan_query" for e in events)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_web_chat_events.py::test_chat_uses_controller_and_is_bounded -v`
Expected: FAIL — no `plan_query` event (old orchestrator path is still active).

- [ ] **Step 3: Delegate `stream_chat_events` to the controller**

In `backend/semantic_layer/web/events.py`, replace the entire `stream_chat_events` function body with a thin delegation (keep `_Provenance` and `_collect_highlight` in the file — they're still imported by tests):

```python
def stream_chat_events(question: str) -> Iterator[dict]:
    """Yield UI events by driving the graph-native controller."""
    from semantic_layer.agent.controller import answer_stream
    yield from answer_stream(question)
```

The `settings` import and the `build_agent` import at the top become unused — remove the `from semantic_layer.agent.build import build_agent` line (the legacy orchestrator stays available via `build.py` for the CLI). Keep `from semantic_layer.config import settings` only if still referenced; otherwise remove it. The `_Provenance` class and `check_numeric_grounding` import remain (used by `test_provenance.py`).

- [ ] **Step 4: Run the web + provenance tests**

Run: `cd backend && .venv/bin/python -m pytest tests/test_web_chat_events.py tests/test_provenance.py tests/test_stream_robustness.py -v`
Expected: the new controller test passes; `test_provenance.py` still passes (it tests `_Provenance` directly). NOTE: `test_stream_robustness.py` monkeypatched the OLD `build_agent` path — if it now fails because the path changed, update those two tests to monkeypatch `semantic_layer.agent.controller.answer_stream` instead (the controller already has its own try/except + bounded behavior; keep one test asserting an answer is always emitted on error and delete the recursion-limit test, since the controller has no recursion limit). Re-run until green.

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/web/events.py backend/tests/test_web_chat_events.py backend/tests/test_stream_robustness.py
git commit -m "feat(web): route chat through the graph-native controller"
```

---

## Self-Review notes

- **Spec coverage:** controller 4-stage drive (Task 5) ✓; parallel `ThreadPoolExecutor` legs (Task 5) ✓; `run_sql/api/doc_leg` workers (Tasks 2–4) ✓; synthesis with `synthesis_model` + grounding reuse (Task 5) ✓; web rewiring + band-aid removal (Task 6) ✓; financials leg so the Max query stays complete (Task 1) ✓; API correlation at synthesis via `api_correlations` (Task 5, passed into `_synthesize`) ✓; UI event-stream compatibility (Tasks 5–6) ✓.
- **Integration-uncertainty points for implementers to validate (mirrors how Plan 2a's bugs were caught):**
  1. `with_structured_output` return type — confirm `.invoke(...)` returns the Pydantic instance (it does for OpenAI chat models); the mock tests pin the contract.
  2. Thread-safety — Neo4j driver is shared/thread-safe; `_run` opens a fresh psycopg/sqlite connection per call; `call_api` uses a module `TestClient`. If sqlite raises a cross-thread error, give each leg its own connection (it already opens per-call). Validate under the live e2e.
  3. `search_documents.invoke({"query": ...})` and `call_api(...)`/`_run(...)` are the real callables — confirm import paths.
- **Bounded cost:** extract(1) + one structured call per leg (+≤1 SQL retry) + synth(1). No `recursion_limit` anywhere on this path.
- **Legacy path:** `build.py`'s deep agent + `ask()` remain for the CLI; only the web path moves to the controller.
- **Reminder:** backend pytest can wipe doc-chunk embeddings — run `python -m semantic_layer.ingest.pipeline` before the live doc/e2e tests.
- **Known follow-ups (out of scope):** streaming partial leg results before synthesis; a second leg wave for genuinely sequential questions; richer financials metric mapping.
```
