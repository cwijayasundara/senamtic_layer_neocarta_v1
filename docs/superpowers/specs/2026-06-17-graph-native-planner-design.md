# Graph-Native Planner + Parallel Legs — Design

**Date:** 2026-06-17
**Status:** Draft (for review)
**Builds on:** [Fiscal-Period Auto-Scoping](2026-06-17-fiscal-period-auto-scoping-design.md),
[Value Layer + Document Context Graph](2026-06-16-value-layer-and-doc-context-graph-design.md),
[Consumption Layer — Grounded Answers](2026-06-17-consumption-layer-grounded-answers-design.md)

## Problem

The current agent answers by letting an LLM **crawl the knowledge graph one node at a
time**. On the cross-source ("Max") question the orchestrator made ~20 sequential
discovery calls — `resolve_value` ×4, `list_tables` ×4, `get_table_schema` ×8,
`get_join_path` ×3 — *before* any subagent did real work, then delegated to SQL/API/doc
subagents **sequentially**. Consequences:

- **Cost/latency**: every discovery step is an LLM round-trip + tool execution. The work
  the graph could answer in a few set-based queries instead costs ~16 LLM turns.
- **Unbounded supersteps**: heavy questions exceeded LangGraph's default 25-superstep cap
  and died mid-run (a band-aid raised it to 100; that only lets an inefficient plan run
  longer).
- **Cross-source gaps**: APIs join to SQL only via a prompt hint ("account_id == customer
  id"); the graph doesn't encode the link, so the LLM has to know it.

The graph already holds everything needed to *plan* the query (tables, columns, FK join
paths, indexed dimension values, business terms, document↔entity↔value↔period bridges). We
should let the graph plan in one shot and reserve the LLM for what it's good at: reading
intent and synthesizing the answer.

## Goals

1. **Fewer LLM round-trips** — collapse discovery into deterministic Cypher (≤3 queries).
2. **Lower wall-clock latency** — run the independent SQL/API/doc legs concurrently.
3. **Cross-source completeness** — encode API↔SQL links in the graph so they're planned,
   not guessed.
4. **Predictable cost ceiling** — a fixed, small number of LLM calls per question
   regardless of complexity.

## Approach: a deterministic controller over a graph-native plan

Replace the free-form ReAct orchestrator (on the web path) with a fixed four-stage flow.
The flow and fan-out are deterministic Python (the **controller**); the per-leg work stays
LLM **subagents** for flexibility (the "hybrid" control flow).

```
question
  → [1] extract_intent(question)      1 LLM call (planner_model)
  → [2] build_plan(intent)            deterministic Cypher, ≤3 queries — 0 LLM
         · batch-resolve all terms → values (table/column/exact spelling)
         · all join paths: fact → each dimension AND fact → bridged API tables
         · context graph → relevant documents + fiscal periods
  → [3] run legs IN PARALLEL          SQL / API / doc concurrently (llm_model, ≤1 each)
  → [4] synthesize(legs)              1 LLM call (synthesis_model) + grounding
```

Only stages 1 and 4 are guaranteed LLM calls; legs add ≤1 each. Total ≈ **5–6 LLM calls,
fixed** — the cost ceiling. The LangGraph `recursion_limit` becomes irrelevant on this
path (no ReAct loop); the existing stream error-wrap stays as a backstop and the cap bump
is reverted.

## Components

| Unit | File | Responsibility | LLM |
|------|------|----------------|-----|
| `extract_intent(question) -> Intent` | `agent/planner.py` (new) | Structured LLM call → terms, fact/metric, group_by, time scope, `needs`, doc/api intents | `planner_model` |
| `build_plan(intent) -> Plan` | `agent/planner.py` (new) | Deterministic Cypher → sql/api/doc legs + highlight set | none |
| `run_sql_leg`, `run_api_leg`, `run_doc_leg` | `agent/legs.py` (new) | Execute one pre-resolved leg; no discovery | ≤1 each (`llm_model`) |
| `answer_stream(question)` | `agent/controller.py` (new) | Drive the 4 stages, fan out legs in threads, emit UI events, synthesize | 1 (`synthesis_model`) |
| `bridge_sources(driver)` | `ingest/bridge.py` (new) | Ingest step: add `SAME_ENTITY` edges API↔SQL | none |
| config additions | `config.py` | `planner_model`, `synthesis_model` (fall back to `llm_model` when unset) | — |

The existing SQL/API/doc subagent prompts and the `run_sql` / `call_api` /
`search_documents` / `get_table_schema` tools are reused as the leg workers, stripped of
discovery tools. The free-form top-level orchestrator (`build_agent`) is retired from the
web path; its graph tools (`resolve_value`, `get_join_path`, `search_catalog`,
`neighbors`, `periods_for_documents`) are superseded by `build_plan`'s Cypher but kept for
the CLI/back-compat.

### Data shapes

```python
Intent = {
  "terms": [str],                      # descriptor tokens: ["EMEA","Cloud","Blackwell","Data Center"]
  "fact": str | None,                  # e.g. "revenue" -> order_line.amount
  "group_by": [str],                   # ["customer","quarter"]
  "time_scope": {"fiscal_year": int, "quarter": str} | None,
  "needs": {"sql": bool, "api": bool, "doc": bool},
  "doc_query": str | None,             # what to look up in documents
  "api_intents": [str],                # ["dgx usage","open tickets"]
}

Plan = {
  "sql_legs": [ {"source", "fact_table", "sql_reference",
                 "filters": [{"table","column","value"}],
                 "joins": [{"on": [colA, colB]}], "scope", "group_by"} ],
  "api_legs": [ {"source", "endpoint", "params"} ],
  "doc_leg": {"doc_query", "candidate_doc_ids": [str], "period"} | None,
  "highlight": [node_id],              # every table/doc in the plan, for the UI graph
}
```

### `build_plan` Cypher (deterministic, ≤3 queries)

1. **Batch value resolution** — terms normalized in Python (reusing `value_indexer.norm`),
   matched in one query:
   ```cypher
   UNWIND $norms AS n
   MATCH (db:Database)-[:HAS_SCHEMA]->(:Schema)-[:HAS_TABLE]->(t:Table)
        -[:HAS_COLUMN]->(c:Column)-[:HAS_VALUE]->(v:Value {norm: n})
   RETURN n, db.name AS source, t.id AS table_id, c.name AS column, v.name AS exact
   ```
2. **All join paths at once** — fact table to every resolved dimension AND to bridged API
   tables, traversing `REFERENCES` and `SAME_ENTITY`. Reuses the same-node guard from
   `get_join_path` (skip pairs where fact == target):
   ```cypher
   MATCH (fact:Table {id:$fact})
   UNWIND $targets AS tgt
   MATCH (tb:Table {id: tgt})
   MATCH p = shortestPath((fact)-[:HAS_COLUMN|REFERENCES|SAME_ENTITY*1..24]-(tb))
   RETURN tgt, [x IN nodes(p) | head(labels(x)) + '|' + x.id] AS nodes ORDER BY length(p)
   ```
3. **Relevant documents + periods** — context graph:
   ```cypher
   MATCH (e:Entity)-[:REFERS_TO]->(v:Value) WHERE v.norm IN $norms
   MATCH (ch:Chunk)-[:MENTIONS]->(e), (d:Document)-[:HAS_CHUNK]->(ch)
   OPTIONAL MATCH (d)-[:COVERS_PERIOD]->(p:Period)
   RETURN DISTINCT d.id AS doc_id, collect(DISTINCT p.key) AS periods
   ```

The doc leg still runs `search_documents` for the actual passages; `candidate_doc_ids`
narrows/justifies them.

## Cross-source bridge edges (`ingest/bridge.py`)

Deterministic step after metadata load. Verified present in the graph: API tables expose
the key (`itsm GET /tickets.account_id`, `dgx GET /usage.account_id`). For each API column
named `account_id`, add:

```cypher
MATCH (ac:Column) WHERE ac.id ENDS WITH '.account_id' AND ac.id STARTS WITH 'col:'
  AND (ac.id CONTAINS ':crm.' OR ac.id CONTAINS ':dgx.' OR ac.id CONTAINS ':itsm.')
MATCH (cust:Column {id:'col:sales_pg.sales.customer.customer_id'})
MERGE (ac)-[:SAME_ENTITY]->(cust)
```

An explicit key-map (`account_id -> sales.customer.customer_id`) keeps it deterministic and
extensible; no value-overlap inference. `build_plan` then folds API tables into the join
graph automatically, so "for these customers, get DGX usage and open tickets" is planned
from the graph rather than from a prompt hint.

## Model tiering

Per-stage model config so the powerful model runs only on the 1–2 reasoning-heavy calls:

```python
planner_model: str | None = None     # e.g. "openai:gpt-5.5" — extract_intent
synthesis_model: str | None = None   # synthesize; defaults to planner_model
# both fall back to llm_model ("openai:gpt-5.4-mini") when unset
```

`extract_intent` uses `planner_model`; legs use `llm_model`; `synthesize` uses
`synthesis_model`. Defaults fall back to `llm_model` so the system runs out of the box; set
`planner_model=openai:gpt-5.5` in `.env` to opt into the stronger planner. **The exact id
must be one the OpenAI account exposes** — `init_chat_model` errors clearly at startup if
not (no silent fallback). A one-shot live call validates the id when wired.

## Parallelism & error handling

Stage 3 fans out legs with a `ThreadPoolExecutor` (Neo4j driver is thread-safe; psycopg
opens a connection per call; the API uses an in-process `TestClient`). A leg that fails
returns a structured `{error}` and synthesis reports partial results — one failed leg never
sinks the whole answer. The controller wraps the run so the UI always receives a final
`answer` event (carried over from the consumption-layer work).

**Leg independence.** A question like "DGX usage *for those top customers*" looks like the
API leg depends on the SQL leg's output. We keep legs independent instead: each leg is
parameterized from the **plan's resolved dimension filters** (EMEA, Cloud, …), not from
another leg's result. The API leg fetches the relevant slice (e.g. EMEA Cloud DGX usage),
and **cross-leg correlation happens at synthesis** using the `SAME_ENTITY` key
(`account_id ↔ customer_id`) — exactly how the earlier live run already behaved (fetch
`/usage?region=EMEA…` → match to the top customers). This preserves single-wave
parallelism. The controller *supports* a second wave for genuinely sequential cases, but
the default plan is one parallel wave + synthesis-time join, and `build_plan` is
responsible for marking any leg that must wait.

## UI compatibility (data flow)

The controller emits the **same event stream** `frontend` already consumes, so
`TracePanel`, graph highlighting, and `AnswerPanel` work unchanged:

- `tool_call`/`tool_result` for `plan_query` (the trace shows the blueprint) and for each
  leg execution;
- a final `answer` event with `sql_runs` / `api_calls` / `doc_citations` / `caveats`
  (reusing `_Provenance` + `check_numeric_grounding`) and `highlight` = every table/doc in
  the plan (richer than today's incidental highlights).

`web/events.py:stream_chat_events` delegates to `controller.answer_stream`; `_Provenance`
and grounding are reused verbatim.

## Testing

- **`bridge_sources`** (neo4j): after ingest, `SAME_ENTITY` exists from each API
  `account_id` to `sales.customer.customer_id`; `get_join_path(order_line, itsm GET /tickets)`
  returns a path through the bridge.
- **`build_plan`** (neo4j, seeded graph): the EMEA/Cloud/Blackwell/Data-Center terms
  resolve to their four dimension tables in one call; the plan's `sql_legs` carry the full
  join chain; `doc_leg.candidate_doc_ids` includes `doc:NVIDIAAn_2026`.
- **`extract_intent`**: structured-output schema validity (mock LLM unit test; one live
  test asserting the four terms + `needs.sql/api/doc` for the Max question).
- **Controller** (offline, fake legs): asserts legs dispatch concurrently, the event-stream
  shape, and a **bounded LLM-call count** (≤6) independent of source count.
- **End-to-end** (live, marked): the Max question fuses 2 DBs + APIs + docs and completes
  in ≤ the bounded call budget, with no `recursion_limit` reliance.

## Build sequence

Two sequenced plans:

1. **Bridge edges** — `ingest/bridge.py` + pipeline wiring + tests. Small, additive,
   independent; lands first and is independently useful.
2. **Planner + controller + parallel legs** — `agent/planner.py`, `agent/legs.py`,
   `agent/controller.py`, config tiering, `web/events.py` rewiring, tests. The larger
   effort; may itself split into (2a) planner/`build_plan` and (2b) controller + legs + UI.

## Out of scope (YAGNI)

- Removing the deepagents subagents entirely (kept as leg workers and for the CLI path).
- Value-overlap / fuzzy auto-bridging — explicit key-map only for now.
- Streaming partial leg results to the UI before all legs finish (nice-to-have; current
  design emits leg results as they complete but synthesizes once).
- New data sources or new bridge keys beyond `account_id ↔ customer_id`.
