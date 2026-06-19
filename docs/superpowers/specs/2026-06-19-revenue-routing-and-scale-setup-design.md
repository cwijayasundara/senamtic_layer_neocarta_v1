# Revenue Fact-Table Routing + Scale-Aware Setup — Design

**Date:** 2026-06-19
**Status:** Draft (for review)
**Builds on:** [Embedding-Based Hybrid Table Retrieval](2026-06-19-embedding-table-retrieval-design.md),
[Scale Agent Performance Harness](2026-06-19-scale-agent-performance-harness-design.md)

## Goal

Two independent fixes surfaced while running the agent at scale:

1. **Revenue fact-table routing.** "Revenue by region/industry/segment" routes to the wrong
   table — `financials.income_statement` (has a literal `revenue` column) and the distractor
   `scale_finance_ops.quarterly_revenue` — instead of the real sales fact
   `sales_pg.sales.order_line` (whose `amount` column *is* sales revenue). Fix so "revenue"
   questions route to `order_line` and answer correctly.
2. **`setup.sh` wipes the scale catalog.** Every `setup.sh` run reseeds the core to 40
   customers and runs a non-scale ingest (`reset=True`), dropping the 1,000 `scale_*` tables
   from Neo4j. Add a `--scale` mode so the scale catalog can be provisioned and persists.

## Fix 1: curated descriptions for core tables

### Root cause

Table embeddings are synthesized from `"<name> — columns: <col names>"`
(`ingest/embeddings.py:101` `_table_embed_text`). SQL extractors do **not** populate
`Table.description` (`sql_extractor.py:84,111` create `Table(id=..., name=t)` only). So
`order_line`'s embed text — `"order_line — columns: line_id, order_id, product_id, quantity,
unit_price, amount"` — never contains "revenue". The query *"revenue by region"* vector-matches
`income_statement` (literal `revenue` column) and the near-miss distractor
`scale_finance_ops.quarterly_revenue`; `order_line` is never routed, and
`select_fact_table` (`routing.py:181`) can only pick from the routed set — so it can't choose
`order_line`. (Confirmed in the eval: `join-revenue-by-industry` and `join-top-customer` fail;
`join-revenue-by-region`/`-segment` "pass" only because the loose `contains` check matches the
keyword in a hedged answer.)

### Approach (chosen)

A small **curated description map** for the ~8 answerable-core tables, folded into the embed
text at `embed_tables` time. Distractors and other tables keep `name + columns`. No LLM pass,
no DDL changes, deterministic, targeted at exactly the tables the failure touches.

- Rejected: LLM-generate descriptions for all 1,072 tables (expensive every ingest, the
  distractors don't need it).
- Rejected: table-level `BusinessTerm` synonym routing (more moving parts; we dropped
  business-term routing from the hybrid path).

### Components

1. **`backend/semantic_layer/ingest/table_descriptions.py`** — a module-level dict
   `TABLE_DESCRIPTIONS: dict[str, str]` mapping core table ids → one-line semantic
   descriptions. The two load-bearing entries disambiguate the revenue trap:
   - `table:sales_pg.sales.order_line` → `"sales revenue line items; amount is the line
     revenue (quantity × unit_price); the source for revenue by region, industry, segment,
     product, period"`
   - `table:financials.main.income_statement` → `"company-level reported quarterly financial
     statements (total revenue, net income); NOT per-order, per-customer, or regional"`
   - Plus concise descriptions for the other core tables (`sales_order`, `customer`,
     `product`, `product_line`, `region`, `country`, `industry`, `segment`, `fiscal_period`)
     to sharpen their matches.
   These are the *only* curated entries; everything else falls back to name+columns.

2. **`_table_embed_text(name, cols, description="")`** (`ingest/embeddings.py:101`) — gains an
   optional `description`; when present it is woven into the embedded text, e.g.
   `"order_line — sales revenue line items; amount is the line revenue … — columns: …"`.
   Signature stays backward-compatible (default `""`).

3. **`embed_tables`** (`ingest/embeddings.py:111`) — looks up
   `TABLE_DESCRIPTIONS.get(row_id, "")` per table and passes it to `_table_embed_text`. No
   Cypher/schema change needed: the description lives in code and is consulted at embed time
   (simpler than writing `Table.description` nodes).

### Data flow

```
embed_tables → per table: text = _table_embed_text(name, cols, TABLE_DESCRIPTIONS.get(id,""))
            → OpenAI embed → Table.embedding → table_embeddings index
query "revenue by region" → vector now matches order_line (its description says "sales revenue")
            → order_line routed → select_fact_table picks it (sales-schema, most FKs)
            → get_join_path order_line→region fills the join → correct SQL
```

### Success criterion

Re-run the eval: `join-revenue-by-industry` and `join-top-customer` answer correctly, and a
direct check that `retrieve_candidate_tables("total revenue by region")` now includes
`table:sales_pg.sales.order_line`. Routing recall/hit-rate on multi-table-join improves; no
regression elsewhere.

## Fix 2: `setup.sh --scale`

### Root cause

`setup.sh:75` seeds the core at baseline (40 customers) and `setup.sh:80/83` runs the
**non-scale** pipeline with `reset=True` → `reset_graph` (`graph/client.py:21`,
`MATCH (n) DETACH DELETE n`) wipes everything, then rebuilds core-only. The 1,000 `scale_*`
tables (still in Postgres) never re-enter Neo4j.

### Approach (chosen)

Add an optional `--scale` flag to `setup.sh`. When set, the seed/ingest steps run the
**existing** scale flow (mirroring the `Makefile` `scale-seed`/`scale-ingest` targets) instead
of the baseline flow. Plain `./setup.sh` is unchanged.

- No `pipeline.py` changes. `scale-ingest` already does `reset=True` and rebuilds the graph
  *with* the scale bundles, so the end state is the full 1,072-table catalog. The wipe is only
  a problem in the non-scale path; `--scale` simply runs the scale path instead.
- Rejected: auto-detect/preserve existing scale (fragile); a `RESET_GRAPH` env knob
  (unnecessary given the above).

### Components

1. **`setup.sh` arg parsing** — accept `--scale` (set `SCALE=true`), update the usage banner,
   error on unknown args. Default `SCALE=false`.
2. **Seed step** — when `SCALE=true`, run `SCALE_MODE=true python -m data.seed_scale` (core at
   scale volume + 1,000 distractor tables) instead of the baseline `seed_postgres`/`seed_sqlite`.
   (Financials/org SQLite seeding still runs — the scaled core is Postgres `sales` only.)
3. **Ingest step** — when `SCALE=true`, run the scale pipeline:
   `SCALE_MODE=true SCHEMA_ROUTING_ENABLED=true FAKE_EMBEDDINGS=true python -m
   semantic_layer.ingest.pipeline` (full pipeline with `with_llm` gated on `HAVE_KEY`, same as
   today). Scale routing needs real table embeddings, which require `OPENAI_API_KEY`; if absent,
   `setup.sh --scale` warns that routing will fall back to keyword (table embeddings are skipped
   without a key, since they run in the LLM stage).

### Success criterion

`./setup.sh --scale` ends with Neo4j holding 1,072 tables (all embedded when a key is present);
`./setup.sh` is byte-for-byte unchanged in behavior.

## Testing

- **Fix 1 unit:** `_table_embed_text("order_line", ["amount"], "sales revenue …")` includes the
  description text; empty description preserves the old `"name — columns: …"` output.
- **Fix 1 unit:** `TABLE_DESCRIPTIONS` contains the two load-bearing keys (`order_line`,
  `income_statement`) and all values are non-empty strings.
- **Fix 1 integration (neo4j+openai):** after `embed_tables` on the ingested graph,
  `retrieve_candidate_tables("What is total revenue by region?")` includes
  `table:sales_pg.sales.order_line`.
- **Fix 1 end-to-end (controller-run):** re-run the answer eval at scale; confirm the two
  failing revenue/multi-join questions now pass and overall accuracy holds/improves.
- **Fix 2:** `bash -n setup.sh` parses; `./setup.sh --help` shows `--scale`; a dry check that
  `--scale` routes to `data.seed_scale` + the scale pipeline command (assert via `set -x` trace
  or a `--dry-run` echo, without running the heavy ingest in CI). The live `./setup.sh --scale`
  is validated manually (it rebuilds the graph).

## Out of scope

- Auto-generating descriptions for non-core tables.
- Re-adding business-term routing to the hybrid path.
- Changing `reset_graph`/`pipeline.py` reset semantics.
- Fixing the multi-table-join *measurement* artifact (intermediate tables not "routed" but
  FK-expanded downstream) — that is a metric question, separate from this correctness fix.
