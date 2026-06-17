# Fiscal-Period Auto-Scoping — Design

**Date:** 2026-06-17
**Status:** Approved
**Builds on:** [Value Layer + Document Context Graph](2026-06-16-value-layer-and-doc-context-graph-design.md)

## Problem

When an answer fuses documents and SQL, the two sides describe different time
windows. The document subagent quotes a specific fiscal quarter (e.g. "Q1 Fiscal
2027 Summary… Revenue $81,615M"); the SQL subagent aggregates `SUM(order_line.amount)`
across **all** order lines because nothing scopes it to a period. Observed in query #5
("what drove Data Center growth, and how big is Blackwell's share"): the doc side
cited Q1; the SQL side returned an all-time 38.81% Blackwell share. The numbers aren't
comparable, and the SQL subagent itself flagged the caveat:

> "The joined path used here does not include a date field… these totals are for all
> available order lines in the schema, not a specific time period."

Root cause: `order_line` reaches the time dimension only via
`order_id → sales_order.fiscal_period_id → fiscal_period`, and neither the orchestrator
nor the SQL subagent is told to use it. Documents carry no period metadata at all
(`doc_loader.py` sets only title/path/num_pages).

## Blocking data finding

Seed sales data and the press releases cover **disjoint** fiscal periods:

| Source | Periods |
|--------|---------|
| `sales.fiscal_period` (`reference.py:60-69`) | FY2024 Q1 → FY2025 Q4 (ends 2025-01-26) |
| `doc:NVIDIAAn_2025` | Q1 FY2026 ("first-quarter… up 73% YoY") |
| `doc:NVIDIAAn_2026` | Q1 FY2027 ("Q1 Fiscal 2027 Summary") |

Naive scoping (`WHERE fiscal_year=2027`) would return zero rows. **Decision: extend the
seed** to add FY2026 Q1–Q4 + FY2027 Q1 so the doc periods overlap real sales rows.

## Approach: a canonical `:Period` node + a Document→Period edge

`fiscal_period` can't ride the `:Value` layer (`fiscal_year` is INTEGER, never
value-indexed; `quarter` is ambiguous across years). Introduce a dedicated node that
SQL periods and document-covered periods converge on — the value-layer bridge, applied
to time.

### New graph elements

- **Node** `:Period {key, fiscal_year, quarter, fiscal_period_id, start_date, end_date, sql_reference}`
  — `key` (e.g. `"FY2027-Q1"`) is the MERGE/match key. `fiscal_period_id` is the FK back
  to `sales.fiscal_period`, or NULL for a doc-covered period with no sales rows.
- **Edge** `(:Document)-[:COVERS_PERIOD]->(:Period)` — the period a press release reports.

Two populations MERGE on `key`, so a doc-covered period attaches to the SQL-backed node
when one exists. `fiscal_period_id IS NOT NULL` ⇒ SQL scoping is possible; NULL ⇒ no sales
rows ⇒ the agent states the scope mismatch instead of returning an empty table.

### Components

1. **`ingest/period_indexer.py` — `index_periods(driver)`** (new). Pure SQL, runs every
   ingest (mirrors `value_indexer.index_values`). `SELECT` from `sales.fiscal_period`,
   MERGE a `:Period {key}` per row with `fiscal_period_id` + dates.

2. **Doc period extraction** in `ingest/doc_graph.py`:
   - `extract_period(doc) -> {fiscal_year, quarter} | None` — regex-first over headline
     chunks: `Q([1-4])\s+(?:Fiscal|FY)\s*'?(\d{2,4})` (normalize FY27→2027), fallback
     `(first|second|third|fourth)-quarter` + nearest 4-digit fiscal year. Deterministic;
     LLM only if regex misses (small-model reliability per prior spec).
   - `link_document_period(driver, doc_id, period)` — `MERGE (p:Period {key})
     MERGE (d)-[:COVERS_PERIOD]->(p)`.

3. **Seed extension** (`reference.py:60-69`) — append FY2026 Q1–Q4 + FY2027 Q1 following
   the late-January fiscal calendar. `generators/sales.py` assigns orders across all
   periods, so new quarters get data on re-seed.

4. **Pipeline wiring** (`pipeline.py:27-60`) — `index_periods` after `index_values`;
   `extract_period` + `link_document_period` in the documents block.

5. **Agent tool `periods_for_documents(doc_ids)`** (graph_tools, next to `neighbors`) —
   returns each doc's `Period`(s) as `{key, fiscal_year, quarter, sql_available}`.

6. **Prompts** (`build.py`):
   - Orchestrator: after resolving docs, call `periods_for_documents`; pass any period to
     the SQL subagent as an explicit scope.
   - SQL subagent: when given a scope, use `get_join_path(order_line, fiscal_period)` (BFS
     already returns the `sales_order`-bridged path) and add
     `WHERE fp.fiscal_year=%s AND fp.quarter=%s`. If 0 rows, re-run all-time and state both
     scopes in the answer.

### Data flow (additions to prior spec)

```
run_ingest:
  reset → SQL metadata → index_values → index_periods (:Period)
        → API metadata → documents (HAS_CHUNK)
                        → extract_period + link_document_period (COVERS_PERIOD)
  with_llm: … entity extraction / bridge / embed (unchanged)
```

## Testing

- `index_periods`: `:Period {key:'FY2025-Q1'}` exists with `fiscal_period_id` set.
- `extract_period`: "Q1 Fiscal 2027"→(2027,'Q1'); "first-quarter…fiscal 2026"→(2026,'Q1').
- `link_document_period`: seeded doc+period → `COVERS_PERIOD`; doc-only period MERGEs onto
  the SQL-backed node.
- `get_join_path(order_line, fiscal_period)` returns the `sales_order`-bridged path.
- End-to-end: query #5 ×3 → scoped Blackwell-DC revenue stable and non-empty for FY2027 Q1.

## Out of scope (YAGNI)

Multiple-period edges per document (Q/Q and Y/Y comparison periods), period inference for
API/SQLite sources, fiscal-calendar config beyond the seed list.
