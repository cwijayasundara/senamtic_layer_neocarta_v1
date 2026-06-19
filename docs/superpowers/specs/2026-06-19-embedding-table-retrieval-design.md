# Embedding-Based Hybrid Table Retrieval — Design

**Date:** 2026-06-19
**Status:** Draft (for review)
**Builds on:** [Graph-Native Planner](2026-06-17-graph-native-planner-design.md),
[Scale Agent Performance Harness](2026-06-19-scale-agent-performance-harness-design.md)

## Goal

Fix the routing-retrieval collapse the scale harness measured: at 1,072 tables the
agent's `retrieve_candidate_tables` returns the wrong tables (routing hit-rate **17.4%**,
recall **0.26**), because retrieval is keyword-only (`search_catalog`) and gets swamped by
~1,000 distractors sharing column names. Replace the table-selection half of retrieval with
**embedding-based vector search**, kept **hybrid** with the existing keyword value/term
matching. Success = a materially higher routing hit-rate/recall on `make eval` vs the 17.4%
baseline, with non-scale behavior unchanged.

## Background: why keyword retrieval fails at scale

`agent/graph_tools.py:279` (`search_catalog`) does case-insensitive substring (`CONTAINS`)
matching over `Table.name`, `Column.name`, `Value.name`, `BusinessTerm.name`. Two failure
modes the harness exposed (`agent/routing.py:33` aggregates these hits):

1. **No semantic match.** The query term *"customers"* does not `CONTAINS`-match the table
   name `customer` (the singular is shorter than the plural term), so the core table is
   never even a candidate.
2. **Distractor pollution.** 1,000 `scale_*` tables share pooled column names (`amount`,
   `region`, `status`). Each per-category Cypher caps at `LIMIT 20`, and the final
   `hits[:limit]` keeps the top 20 — so distractors crowd out the core tables, and
   near-miss distractors (`scale_finance_ops.quarterly_revenue`,
   `scale_marketing.campaign_revenue`) out-rank the real `sales` tables.

The LLM ranker (`rank_tables`) can only pick from what retrieval feeds it, so the bottleneck
is **retrieval, not ranking**.

## Decisive constraint: NeoCarta embeds only `description`

The natural move — "use NeoCarta's metadata embedder" — does not work here. NeoCarta's
`OpenAIEmbeddingsConnector` (`embed_metadata_nodes`, `ingest/embeddings.py:94`) embeds nodes
only `WHERE n.description IS NOT NULL AND size(n.description) > 0`, embedding the
`description` text. Our tables are introspected from Postgres/SQLite with **names only — no
descriptions** — so the connector embeds nothing useful for tables and would also create a
colliding `table_vector_index`.

Therefore we **own the table embedding** ourselves, exactly as the repo already owns chunk
embeddings (`embed_chunks` + `_ensure_chunk_vector_index`, `ingest/embeddings.py:30,73`):
synthesize a rich text per table from what we *do* have (qualified name + column names) and
embed that into a vector index we control by a stable name.

## Decisions locked (2026-06-19)

- **Own the table embedding** (synthesized name+columns text), not NeoCarta's
  description-only connector. Drop the `embed_metadata_nodes` NeoCarta call.
- **Always embed tables for real**, even when `fake_embeddings` is on. `fake_embeddings`
  continues to fake only *chunk* vectors (doc-RAG, not under test). Table embedding is what
  routing — the thing we measure — depends on, so it must be real. Cost at ~1,072 tables ≈
  ~17 batched embed calls (~1¢).
- **Hybrid retrieval**: vector table-similarity UNION the existing keyword Value +
  BusinessTerm hits. Keep exact data-value routing ('EMEA', 'Blackwell', 'Data Center')
  that embeddings cannot do.
- **Graceful fallback**: if the table vector index is missing/empty, fall back to today's
  keyword path. No hard dependency; backward-compatible.

## Components

### 1. `embed_tables(driver)` — `ingest/embeddings.py`

- Query each table with its columns:
  `MATCH (t:Table) OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column) RETURN t.id, t.name, collect(c.name) AS cols`.
- Synthesize the embed text per table: `f"{t.name} — columns: {', '.join(cols)}"` (the
  qualified `id` already encodes source/schema; the name + column list is the discriminating
  signal).
- Embed in batches via the real OpenAI client (`get_openai_client().embeddings.create(model=settings.embedding_model, input=[...], dimensions=settings.embedding_dimensions)`),
  write to `Table.embedding` with `db.create.setNodeVectorProperty`, mirroring the real path
  of `embed_chunks`.
- **Always real** — no `fake_embeddings` branch (unlike `embed_chunks`).
- Call `_ensure_table_vector_index(driver)` to create a vector index named **`table_embeddings`**
  on `Table.embedding` (1536 dims, cosine), mirroring `_ensure_chunk_vector_index`; drop any
  pre-existing NeoCarta `table_vector_index` first to avoid a duplicate-index error.

### 2. Pipeline wiring — `ingest/pipeline.py`

In `_run_llm_stages`, **replace** the `embed_metadata_nodes(driver)` call with
`embed_tables(driver)`. `embed_chunks(driver)` is unchanged (still honors `fake_embeddings`).

### 3. Query-embedding helper

Extract the question→vector embedding (currently inline in `agent/doc_tools.py:18`) into one
small reusable helper so `doc_tools` and `routing` share it (DRY). Location:
`ingest/embeddings.py` as `embed_query(text: str) -> list[float]`. `doc_tools.search_documents`
is updated to call it.

### 4. Hybrid `retrieve_candidate_tables` — `agent/routing.py`

```
def retrieve_candidate_tables(question, k_ret=20):
    vector_hits = _vector_table_hits(question, k=settings.schema_routing_k_vec)   # [{table_id, score(cosine)}]
    keyword_hits = _keyword_value_term_hits(question)                            # value + business_term hits from search_catalog
    merge by table_id (max score per source-class, vector preferred on ties),
    sort desc, return top k_ret as [{table_id, score}]
```

- `_vector_table_hits`: embed the question (helper §3) and
  `CALL db.index.vector.queryNodes('table_embeddings', $k, $vec) YIELD node, score RETURN node.id, score`.
  On a `ClientError`/missing-index (index absent or empty), return `[]` so the caller falls
  back to keyword.
- `_keyword_value_term_hits`: reuse `search_catalog` but keep only `kind in {value, business_term}`
  hits (the exact-value routing embeddings can't replace). Table/column keyword hits are
  superseded by the vector path.
- **Fallback:** if `vector_hits` is empty (no index / no embeddings), use the full current
  `search_catalog` aggregation so behavior degrades to today's keyword path rather than
  returning nothing.
- `rank_tables` and `route_tables` are unchanged — retrieval still returns `[{table_id, score}]`
  and the LLM ranker does the precision trimming.

### 5. Config — `config.py`

Add `schema_routing_k_vec: int = 30` (vector top-k feeding the merge; generous for recall,
the LLM ranker trims). `schema_routing_k_ret` / `k_rank` unchanged.

## Data flow

```
ingest (real table embeds, always):
  embed_tables → Table.embedding + `table_embeddings` vector index (1536, cosine)

query time:
  retrieve_candidate_tables(question)
     ├─ _vector_table_hits → embed_query → queryNodes('table_embeddings', k_vec) → [{table_id, cosine}]
     ├─ _keyword_value_term_hits → search_catalog → keep value/business_term hits
     └─ merge/dedup/cap k_ret  ─(empty vector? → keyword fallback)
  → rank_tables (LLM, unchanged) → route_tables
```

## Testing

- **`embed_tables` unit (neo4j+openai markers):** after `embed_tables`, every `Table` has an
  `embedding`, and the `table_embeddings` index exists.
- **`retrieve_candidate_tables` unit (mock):** monkeypatch `_vector_table_hits` and
  `search_catalog` (mirroring `test_routing.py:7`), assert vector + keyword hits merge,
  dedup, and cap at `k_ret`; assert keyword **fallback** fires when vector hits are empty.
- **Integration (neo4j+openai, ingested graph):** `retrieve_candidate_tables("How many
  customers are there?")` includes `table:sales_pg.sales.customer` — the exact case keyword
  retrieval missed.
- **End-to-end:** re-run `make eval` against the scaled graph (rebuilt so tables are embedded
  for real) and compare routing hit-rate/recall to the 17.4% baseline. Materially higher is
  the success criterion. (Per project policy the full pytest suite wipes doc embeddings — run
  focused test files; this is documented in the harness spec.)

## Error handling & backward compatibility

- Missing/empty `table_embeddings` index → `_vector_table_hits` returns `[]` → keyword
  fallback. A graph ingested before this feature still routes (at today's quality) until
  re-ingested.
- Non-scale ingest is unaffected in shape: `embed_tables` replaces `embed_metadata_nodes` and
  embeds the modest core-table set; `embed_chunks`/`fake_embeddings` semantics for chunks are
  untouched.
- `OPENAI_API_KEY` absent → `embed_tables` raises at ingest (same as the existing real
  embedding paths); this is an ingest-time failure, not a silent degrade.

## Out of scope

- Column-level embeddings / column vector search (tables suffice for routing; YAGNI).
- Changing `rank_tables` / the LLM ranker.
- Re-tuning `k_ret`/`k_rank` (the harness can inform that separately).
- A fake (deterministic) table-embedding mode — rejected: it would make routing scores
  meaningless.
- Auto-generating real `Table.description` text (the synthesized name+columns embed text is
  sufficient and avoids an LLM pass).

## Implementation notes (post-merge)

Two intentional refinements during implementation, for the next reader:

- **Value routing queries the value layer directly, not `search_catalog`.** `search_catalog`
  returns a concatenated `hits[:limit]`, so at ~1,000 tables the column/table keyword hits
  fill the cap and the value hits are truncated out. `_keyword_value_hits` therefore runs a
  direct `(Column)-[:HAS_VALUE]->(Value)` Cypher. A side effect: **BusinessTerm** hits are no
  longer unioned into the vector-path merge (they survive only via the keyword `_keyword_fallback`
  used for embedding-less graphs). The "value/business_term" wording above describes the original
  intent; the shipped hybrid path is vector ∪ exact-value.
- **Value-matched tables are pinned at score 1.0 in the merge** (`setdefault(tid, 1.0)`). Since
  cosine scores are ≤ 1.0, this guarantees value tables enter the candidate set at/above vector
  hits — inclusion is the goal, and the LLM ranker (`rank_tables`) re-scores afterward, so exact
  ordering within the candidate set does not affect the final selection.

**Measured result (live, 1,072-table scaled graph):** routing hit-rate 17.4% → 56.5%, recall
0.26 → 0.68, precision 0.30 → 0.66. Remaining weak spots — multi-table-join recall 0.31 and
near-miss 0.25 — are future tuning (k_vec/k_ret, FK-neighbor expansion), not defects.
