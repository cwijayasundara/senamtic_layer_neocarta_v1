# Scale Agent Performance Harness — Design

**Date:** 2026-06-19
**Status:** Draft (for review)
**Builds on:** [NeoCarta-Local Semantic Layer](2026-06-16-neocarta-local-semantic-layer-design.md),
[Consumption Layer — Grounded Answers](2026-06-17-consumption-layer-grounded-answers-design.md),
[Graph-Native Planner](2026-06-17-graph-native-planner-design.md)

## Goal

Prove that the **agents built on top of the semantic layer keep answering correctly and
fast when the catalog gets large and noisy** — thousands of tables and dozens of APIs,
not the ~16 tables / 4 APIs the app ships with today.

The deliverable is a **repeatable agent-performance harness**, not just bulk data. It has
two halves:

1. A **scale-data generator** that grows the catalog to a configurable size (default
   ~1,000 tables / ~50 APIs) by adding realistic *distractor* domains around a
   high-volume *answerable core*.
2. A **golden eval set + scorer** that runs the real agent (`agent/build.py`) end-to-end
   over the scaled catalog and reports routing quality, answer correctness, and latency —
   with a baseline-vs-scaled comparison so regressions are visible as numbers.

This directly exercises the schema-routing machinery (`agent/routing.py`,
`schema_routing_enabled`) that was built for exactly this "many tables" regime but ships
disabled by default.

## Scope decisions (locked 2026-06-19)

- **Primary intent:** verify agent performance (routing + answer correctness + latency) at
  scale. The data is a means; the scorer is the product.
- **Distractors are catalog-only:** generated distractor tables are **empty** (metadata
  only). Only the coherent answerable core (sales / finance / org / CRM) holds real rows.
- **Stay on the real ingest path** (Approach 1, below): distractors are materialized as
  real Postgres tables and real synthetic OpenAPI specs so the existing extractors,
  Neo4j load, embedding, and routing all run unchanged and under real load.
- **Default scale target:** ~1,000 tables / ~50 APIs, fully parameterized.
- **Embeddings default to deterministic fakes** in scale mode for cost-free iteration; a
  flag switches to real OpenAI embeddings for fidelity runs.
- **Golden set starts at ~40 questions** across the taxonomy in §4.

## Approach: real distractors on the real path + a golden scorer

The app's ingest **introspects live sources** — `sql_extractor.py` reads Postgres
`information_schema`; `api_extractor.py` reads FastAPI-generated OpenAPI JSON. Two
injection strategies were considered:

- **(Chosen) Approach 1 — real empty distractor tables + synthetic API specs.** Create
  ~1,000 empty Postgres tables under generated schemas and feed ~46 synthetic OpenAPI
  specs through the existing API extractor, alongside the 4 real live APIs and a
  scaled-up core. Empty tables are cheap to create and introspect, but loading and
  embedding their metadata is real work — which *is* the ingest/routing test. Keeps the
  whole route → SQL → execute → answer path live for the answerable core.
- **(Rejected) Approach 2 — synthetic `SchemaBundle`s straight into Neo4j.** Fastest, but
  bypasses DB introspection and yields non-queryable distractors; lower fidelity on the
  exact ingest path we want to stress.
- **(Rejected) Approach 3 — light rows in every table.** Contradicts the catalog-only
  decision; slowest and heaviest to seed and ingest.

## Components

### 1. Scale-data generator — `backend/data/generators/scale_catalog.py`

Pure, deterministic, seed-based (no I/O). Parameterized by `n_tables`, `n_apis`, `seed`.

- Emits synthetic business **domains** (e.g. `hr`, `logistics`, `marketing`, `iot`,
  `supply_chain`, `finance_ops`, `support_ops`). Each domain is a *cluster* of tables with
  realistic column names/types, intra-domain foreign keys, and a few cross-domain FKs —
  so routing faces *plausible* distractors, not random noise.
- Returns a declarative catalog spec: a list of table definitions
  (`schema`, `table`, `columns[{name,type,is_pk,is_fk,ref}]`) and a list of API
  definitions (`source`, `endpoints[{verb,path,response_model{fields}}]`).
- Includes a small number of **near-miss** tables whose names/columns deliberately collide
  with answerable-core keywords (e.g. a `marketing.campaign_revenue` next to the real
  `sales.order_line`) to make routing discriminate, not keyword-match.
- Unit-testable in isolation: given a seed and size, asserts table/column/FK counts and
  determinism.

### 2. Distractor seeder — `backend/data/seed_scale.py`

- Takes the catalog spec and `CREATE SCHEMA` / `CREATE TABLE` (no rows) in Postgres, in
  FK-valid order, idempotently. Tables land under `scale_<domain>` schemas, segregated
  from the real `sales` schema.
- Reuses the existing connection/config plumbing in `data/seed_postgres.py`.
- A teardown path (`DROP SCHEMA … CASCADE` for `scale_*`) so scale runs are reversible and
  the dev DB can be reset to baseline.

### 3. Answerable-core scale-up

The existing generators already accept size parameters
(`generate_sales(seed, n_customers, n_orders)`); scale mode raises them:

- `sales`: `n_customers` → `scale_core_customers` (default ~5,000),
  `n_orders` → `scale_core_orders` (default ~50,000); order_lines follow.
- `financials` / `org` / API data scale proportionally.

These remain the **ground-truth** tables the agent must find and query. High row volume
makes aggregations and join latency realistic.

### 4. Pipeline integration — `ingest/pipeline.py` + `config.py`

A `scale` branch in `run_ingest()`:

1. Seed the scaled core (high row volume).
2. Materialize empty distractor tables (`seed_scale.py`).
3. Extract metadata for core + distractors via the existing `sql_extractor`.
4. Feed the 4 real live OpenAPI specs **plus** ~46 synthetic specs into `extract_all_apis`
   (the spec getter is extended to yield synthetic specs from the catalog).
5. Load all bundles into Neo4j, index values/periods, bridge, embed — unchanged code paths.

New `config.py` knobs (all defaulted so non-scale runs are unaffected):

| Knob | Default | Purpose |
|---|---|---|
| `scale_mode` | `False` | Master switch; when on, **auto-enables `schema_routing`** |
| `scale_n_tables` | `1000` | Distractor table target |
| `scale_n_apis` | `50` | Total API target (4 real + synthetic) |
| `scale_core_customers` | `5000` | Answerable-core row volume |
| `scale_core_orders` | `50000` | Answerable-core row volume |
| `fake_embeddings` | `True` (scale mode) | Deterministic hash vectors; no OpenAI spend |

Scale mode auto-enables `schema_routing_enabled` because, without routing, the agent
receives an undifferentiated thousand-table catalog and the test is meaningless.

### 5. Golden eval set — `backend/eval/golden_questions.json`

~40 questions. Each entry:

```jsonc
{
  "id": "agg-orders-by-region",
  "question": "What was total order revenue by region in FY2025?",
  "expected_sources": ["sales_pg"],
  "expected_tables": ["sales.sales_order", "sales.order_line", "sales.region"],
  "expected_join_path": ["order_line→sales_order→customer→region"],
  "answer_check": { "type": "contains", "values": ["region", "FY2025"] },
  "category": "multi-table-join"
}
```

Taxonomy (mirrors what the agents must handle):

- **single-table aggregation** — sum/count/avg over one core table.
- **multi-table join** — 2–4 table joins within a source.
- **cross-source bridge** — SQL ↔ API joins via `SAME_ENTITY` (account_id ↔ customer_id).
- **document RAG** — answers grounded in the NVIDIA PDFs.
- **period filtering** — fiscal-period auto-scoping.
- **near-miss routing** — answer lives in the core but keywords collide with distractor
  domains; passes only if routing ignores the distractors.

Expected tables/answers are derived from the **core schema + `query_log_sales.json` join
patterns**, so they are computable ground truth, not guesses. The golden set targets the
core only (distractors are empty and intentionally unanswerable).

### 6. Eval runner + scorer — `backend/eval/run_eval.py`

Runs each golden question through the **real agent** and reports:

- **Routing quality** — precision / recall@k of selected tables vs `expected_tables`,
  scored directly off `retrieve_candidate_tables` + `rank_tables`. This is the headline
  metric (the "LinkedIn 9%→48%" number).
- **Answer correctness** — `answer_check` assertions (exact / contains / numeric
  tolerance), with an LLM-judge fallback for free-text answers.
- **Latency** — per-stage timing (route / SQL-gen / execute / total), p50 / p95.
- **Output** — a scorecard JSON plus a human-readable summary table; supports
  baseline (small catalog) vs scaled runs so degradation is a visible delta.

The runner can score routing **without** full answer generation (cheaper) or run the full
agent end-to-end, selected by flag.

### 7. Load / concurrency harness — `backend/eval/load_test.py` (lighter, optional)

Fires K concurrent `/chat` requests drawn from the golden set; reports throughput and
p50/p95 under contention. Also exercises the rate-limiter and the Redis query cache.

### 8. Wiring — `Makefile`

- `make scale-seed` — seed scaled core + distractor tables.
- `make scale-ingest` — run ingest in scale mode.
- `make eval` — run the scorer against the current (scaled) graph.
- `make eval-baseline` — run the scorer against the default small catalog for comparison.
- `make scale-teardown` — drop `scale_*` schemas, restore baseline.

Everything is seed-deterministic, so runs are reproducible and diffable.

## Data flow

```
scale_catalog.generate(seed, n_tables, n_apis)
      │  declarative catalog spec (tables + API defs)
      ├─► seed_scale.py ──► empty Postgres tables (scale_* schemas)
      │                         │
generators (core, scaled) ──► real rows (sales/finance/org/CRM)
      │                         │
      └─► synthetic OpenAPI specs ─┐
                                   ▼
                ingest/pipeline.py (scale branch)
        sql_extractor + api_extractor → SchemaBundles
                → Neo4j load + value/period index + bridge + embed (fake|real)
                                   ▼
                       Neo4j graph (schema_routing on)
                                   ▼
   eval/run_eval.py ── golden_questions.json ──► agent/build.py
        → routing precision/recall@k, answer correctness, latency
        → scorecard.json + summary (baseline vs scaled)
```

## Testing

- **Generator unit tests** — determinism (same seed → same catalog), count invariants,
  FK validity, presence of near-miss tables.
- **Seeder test** — creates and tears down a small distractor set against a test DB;
  asserts `sql_extractor` introspects the expected table/column/FK counts.
- **Scorer test** — runs the eval runner over a 2–3 question fixture set with a stubbed
  agent to verify metric computation (precision/recall, assertion checks, latency capture)
  without LLM cost.
- **Smoke** — `make scale-ingest` with a small override (`scale_n_tables=50`) completes and
  `make eval` produces a scorecard, in CI-affordable time.

## Error handling & guardrails

- Distractor seeding is idempotent and namespaced (`scale_*`); never touches the real
  `sales` schema. Teardown is explicit.
- `fake_embeddings` keeps iteration free; real embeddings are opt-in and the runner logs
  the embedding mode in the scorecard so results aren't silently compared across modes.
- The scorer **never** treats a missing/empty result as a pass — an unanswerable routing
  to a distractor is recorded as a routing failure, not a silent skip.
- Per the project memory: running backend pytest wipes doc embeddings, so `make eval`
  documents that a `make ingest` (or `make scale-ingest`) must precede doc-RAG questions.

## Out of scope

- Real external data sources (GCP/BigQuery) — unchanged from prior scope decisions.
- MCP servers — the agent keeps its deepagents direct-tool orchestration.
- UI changes — the harness is CLI/JSON; the existing chat UI is untouched.
- Auto-tuning routing knobs (`k_ret`, `k_rank`) — the harness *measures*; tuning is a
  follow-up informed by its output.
