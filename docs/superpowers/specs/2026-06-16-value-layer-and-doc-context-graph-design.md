# Value Layer + Document Context Graph — Design

**Date:** 2026-06-16
**Status:** Approved

## Problem

Two related defects, one root cause: real-world *things* exist in the data but not as
nodes in the knowledge graph.

1. **Query routing is unreliable.** The catalog stores metadata (table/column/term
   names), not row values. A question filtered by data values — "EMEA Cloud customers
   who bought Blackwell Data Center products" — gives the orchestrator no graph signal:
   `search_catalog` can't match "EMEA"/"Cloud"/"Blackwell"/"Data Center" (they are rows,
   not names). The agent must remember to call `resolve_value` and decompose compound
   phrases; with a small model this is unreliable — observed failures include collapsing
   "Blackwell Data Center" into one `product.name ILIKE '%Blackwell Data Center%'` and
   dropping the industry filter entirely.

2. **Documents are not a context graph.** PDFs ingest as `Document-[:HAS_CHUNK]->Chunk`
   only. Entity extraction is sparse (capped at 40 chunks), isolated, and never bridged
   to the structured catalog. The PDFs show up as effectively single nodes.

## Approach: a shared canonical Value layer

Introduce one node type, `:Value`, that both the SQL dimensions and the document
entities connect to. A dimension row value and a document entity that name the same
real-world thing converge on the **same** `:Value` node — this is the NeoCarta bridge.

### New graph elements

- **Node** `:Value {name, norm}` — `norm` is the lowercased, whitespace-collapsed key
  used for MERGE and matching; `name` is the canonical stored spelling.
- **Edge** `(:Column)-[:HAS_VALUE]->(:Value)` — a SQL column contains this value.
- **Edge** `(:Entity)-[:REFERS_TO]->(:Value)` — a document entity denotes this value.
- Richer `(:Chunk)-[:MENTIONS]->(:Entity {name, label, norm})` over **all** chunks.

### Components

1. **`ingest/value_indexer.py` — `index_values(driver, max_distinct=50)`**
   For each text column (type matches char/text/string) in the POSTGRESQL/SQLITE
   sources, `SELECT DISTINCT` the column; if it has ≤ `max_distinct` non-null values,
   MERGE a `:Value {norm}` per value and link `(:Column)-[:HAS_VALUE]->(:Value)`.
   Pure SQL — runs on every ingest, including `with_llm=False`. Identifiers are
   validated against `^[A-Za-z_][A-Za-z0-9_.]*$`; values are read via the existing
   read-only `_run`. High-cardinality columns are skipped (error/over-limit → skip).

2. **`search_catalog` value branch.** Add a query matching terms against `Value.name`,
   returning `{kind:"value", id, name (exact stored value), table_id, column}`. The
   agent's first call now surfaces "Blackwell→architecture", "Data Center→segment",
   "Cloud→industry (Cloud Service Provider)", "EMEA→region" without extra discipline.
   `resolve_value` stays as a backup.

3. **`ingest/doc_graph.py`**
   - `load_entities(driver, chunk_id, entities)` — MERGE `:Entity {norm}` (set
     name/label), MERGE `(:Chunk)-[:MENTIONS]->(:Entity)`.
   - `bridge_entities_to_values(driver)` — `MATCH (e:Entity),(v:Value) WHERE
     e.norm=v.norm MERGE (e)-[:REFERS_TO]->(v)`.
   Pipeline's LLM stage extracts entities for **all** chunks (cap lifted from 40),
   loads them, then bridges. Entities with no catalog match (e.g. "Spectrum-X",
   "AI factories") remain as connected nodes — the PDFs become a real graph.
   *No entity↔entity relationships (out of scope; not GraphRAG).*

4. **`neighbors(name)` agent tool** (graph_tools). Given a value/entity name, return its
   cross-source neighborhood: catalog columns/tables that `HAS_VALUE` it, and
   documents/chunks that `MENTION` it (via a bridged entity or direct name match).
   Lets the agent hop doc↔catalog: "press releases credit Blackwell for Data Center
   growth, and we recorded $122.6M of Blackwell Data Center sales." Wired into the
   orchestrator tool list + prompt.

### Data flow

```
run_ingest:
  reset → SQL metadata (bundles) → index_values (HAS_VALUE)
        → API metadata → documents (HAS_CHUNK)
  with_llm: glossary → extract_entities(all chunks) → load_entities (MENTIONS)
          → bridge_entities_to_values (REFERS_TO) → embed_chunks → embed_metadata
```

## Testing

- `index_values`: `:Value` nodes + `HAS_VALUE` exist for region/industry/segment/
  architecture; e.g. a Value `norm='blackwell'` linked to `architecture.name`.
- `search_catalog`: term "Blackwell" returns a `kind:"value"` hit pointing at the
  architecture table with exact spelling; "Data Center"→segment; "Cloud"→industry.
- `bridge_entities_to_values`: seed an `:Entity` and `:Value` with matching `norm`,
  run the bridge, assert `REFERS_TO` exists (no LLM needed).
- `neighbors`: `neighbors("Blackwell")` returns the architecture column in `catalog`;
  with a seeded bridged entity, returns the mentioning document.
- Reliability: re-ingest with embeddings; run query #4 ×3 — must apply all four
  dimension filters with correct ground-truth numbers (Flowers Q1 195,327.48; Reid
  Q1/Q3/Q4 1,285,857.49 / 1,013,372.91 / 1,497,014.34). Restore embeddings after the
  suite (the test suite re-ingests without embeddings).

## Out of scope (YAGNI)

Entity↔entity relationship extraction, GraphRAG communities/summaries, value indexing
for high-cardinality columns, any UI changes beyond what the richer graph renders.
