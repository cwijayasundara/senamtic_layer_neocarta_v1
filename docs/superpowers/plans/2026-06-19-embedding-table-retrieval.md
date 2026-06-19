# Embedding-Based Hybrid Table Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `retrieve_candidate_tables` use embedding-based vector search over table metadata (hybrid with keyword value-routing) so the agent finds the right tables among ~1,000 distractors, lifting the routing hit-rate above the measured 17.4% baseline.

**Architecture:** We own table embeddings (the repo already owns chunk embeddings): a new `embed_tables` synthesizes `"name — columns: …"` text per `Table`, embeds it for real, and writes `Table.embedding` + a `table_embeddings` vector index. `retrieve_candidate_tables` becomes hybrid — vector top-k over `table_embeddings` UNION exact value-routing keyword hits — with a graceful fallback to today's keyword path when no table embeddings exist.

**Tech Stack:** Python 3.11+, Neo4j vector index (`db.index.vector.queryNodes`), OpenAI `text-embedding-3-small` (1536 dims), pytest (markers: `neo4j`, `openai`).

## Global Constraints

- Python `>=3.11`; run all commands from `backend/`. Run only the FOCUSED test files named in each task — the full backend `pytest` wipes document embeddings by project policy.
- Vector index is named **`table_embeddings`** on `Table.embedding`, **1536 dims, cosine** — created exactly like the existing `chunk_embeddings` index (`ingest/embeddings.py:73`). Drop any NeoCarta `table_vector_index` first.
- Table embeddings are **always real** (embed_tables ignores `settings.fake_embeddings`); `fake_embeddings` continues to fake only *chunk* vectors. Routing depends on table embeddings, so they must be real.
- Query embedding uses the existing pattern: `get_openai_client().embeddings.create(model=settings.embedding_model, input=[text], dimensions=settings.embedding_dimensions).data[0].embedding`.
- Hybrid retrieval = vector table hits UNION keyword **value** hits (exact data-value routing like 'EMEA'/'Blackwell'); fall back to the current full keyword aggregation when vector hits are empty (no/empty index). `rank_tables` and `route_tables` are unchanged.
- `retrieve_candidate_tables(question, k_ret=20) -> list[dict]` keeps its signature and `[{table_id, score}]` return shape (so `route_tables` and the LLM ranker are untouched).
- Commit after every task with the message shown in its final step.

---

### Task 1: Config knob `schema_routing_k_vec`

**Files:**
- Modify: `backend/semantic_layer/config.py:50` (after `schema_routing_max_targets`)
- Test: `backend/tests/test_routing_config.py`

**Interfaces:**
- Produces: `settings.schema_routing_k_vec: int` (default 30), env `SCHEMA_ROUTING_K_VEC`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_routing_config.py
from semantic_layer.config import Settings


def test_k_vec_default():
    assert Settings().schema_routing_k_vec == 30


def test_k_vec_reads_env(monkeypatch):
    monkeypatch.setenv("SCHEMA_ROUTING_K_VEC", "12")
    assert Settings().schema_routing_k_vec == 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_routing_config.py -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'schema_routing_k_vec'`

- [ ] **Step 3: Write minimal implementation**

In `backend/semantic_layer/config.py`, immediately after the line
`schema_routing_max_targets: int = 8` (line 50), insert:

```python
    # Vector breadth for embedding-based table retrieval: how many tables the
    # vector search returns before the keyword value-hits are unioned in.
    schema_routing_k_vec: int = 30
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_routing_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/config.py backend/tests/test_routing_config.py
git commit -m "feat(routing): schema_routing_k_vec config knob"
```

---

### Task 2: `embed_query` helper + DRY refactor of doc_tools

**Files:**
- Modify: `backend/semantic_layer/ingest/embeddings.py` (add `embed_query`)
- Modify: `backend/semantic_layer/agent/doc_tools.py:18-21` (use `embed_query`)
- Test: `backend/tests/test_embed_query.py`

**Interfaces:**
- Produces: `embed_query(text: str) -> list[float]` in `semantic_layer.ingest.embeddings`.
- Consumes: `get_openai_client` (`ingest.llm`), `settings.embedding_model/embedding_dimensions`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_embed_query.py
from semantic_layer.ingest import embeddings


class _FakeEmbeddings:
    def create(self, model, input, dimensions):
        # one vector per input string; deterministic, no network
        data = [type("E", (), {"embedding": [float(len(s))] * dimensions})() for s in input]
        return type("R", (), {"data": data})()


class _FakeClient:
    embeddings = _FakeEmbeddings()


def test_embed_query_returns_single_vector(monkeypatch):
    monkeypatch.setattr(embeddings, "get_openai_client", lambda: _FakeClient())
    monkeypatch.setattr(embeddings.settings, "embedding_dimensions", 4)
    out = embeddings.embed_query("hello")
    assert out == [5.0, 5.0, 5.0, 5.0]   # len("hello") == 5, dim 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_embed_query.py -v`
Expected: FAIL with `AttributeError: module 'semantic_layer.ingest.embeddings' has no attribute 'embed_query'`

- [ ] **Step 3: Add `embed_query` to `embeddings.py`**

In `backend/semantic_layer/ingest/embeddings.py`, add after the `fake_vector`
function (after line 27):

```python
def embed_query(text: str) -> list[float]:
    """Embed a single query string to a vector for query-time vector search.
    Shared by document and table retrieval so the call lives in one place."""
    return get_openai_client().embeddings.create(
        model=settings.embedding_model, input=[text],
        dimensions=settings.embedding_dimensions,
    ).data[0].embedding
```

- [ ] **Step 4: Refactor `doc_tools.py` to use it**

In `backend/semantic_layer/agent/doc_tools.py`, replace the inline embed (lines 18-21):

```python
    vec = get_openai_client().embeddings.create(
        model=settings.embedding_model, input=[query],
        dimensions=settings.embedding_dimensions,
    ).data[0].embedding
```

with:

```python
    vec = embed_query(query)
```

Update the imports at the top of `doc_tools.py`: replace
`from semantic_layer.ingest.llm import get_openai_client` with
`from semantic_layer.ingest.embeddings import embed_query`. (`get_openai_client` is no
longer used in this file; `settings` is still used for `neo4j_database`.)

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_embed_query.py -v`
Expected: PASS (1 test)

- [ ] **Step 6: Commit**

```bash
git add backend/semantic_layer/ingest/embeddings.py backend/semantic_layer/agent/doc_tools.py backend/tests/test_embed_query.py
git commit -m "refactor(embeddings): extract shared embed_query; doc_tools uses it"
```

---

### Task 3: `embed_tables` + `table_embeddings` vector index

**Files:**
- Modify: `backend/semantic_layer/ingest/embeddings.py` (add `_table_embed_text`, `embed_tables`, `_ensure_table_vector_index`)
- Test: `backend/tests/test_embed_tables.py`

**Interfaces:**
- Produces: `_table_embed_text(name: str, cols: list[str]) -> str`; `embed_tables(driver, batch: int = 64) -> None`; `_ensure_table_vector_index(driver) -> None`. Writes `Table.embedding` and a `table_embeddings` vector index (1536, cosine).
- Consumes: `get_openai_client`, `settings`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_embed_tables.py
import pytest

from semantic_layer.ingest import embeddings
from semantic_layer.ingest.embeddings import _table_embed_text


def test_table_embed_text_includes_name_and_columns():
    assert _table_embed_text("order_line", ["line_id", "amount"]) == \
        "order_line — columns: line_id, amount"


def test_table_embed_text_without_columns():
    assert _table_embed_text("region", []) == "region"


@pytest.mark.neo4j
@pytest.mark.openai
def test_embed_tables_writes_embeddings_and_index(ingested_graph):
    driver = ingested_graph
    embeddings.embed_tables(driver)
    from semantic_layer.config import settings
    with driver.session(database=settings.neo4j_database) as s:
        missing = s.run(
            "MATCH (t:Table) WHERE t.embedding IS NULL RETURN count(t) AS c"
        ).single()["c"]
        idx = s.run(
            "SHOW VECTOR INDEXES YIELD name WHERE name = 'table_embeddings' RETURN count(*) AS c"
        ).single()["c"]
    assert missing == 0
    assert idx == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_embed_tables.py -v`
Expected: FAIL with `ImportError: cannot import name '_table_embed_text'` (the two pure
tests fail at import; the marked test skips without Neo4j/OpenAI).

- [ ] **Step 3: Write the implementation**

In `backend/semantic_layer/ingest/embeddings.py`, add after `_ensure_chunk_vector_index`
(after line 91):

```python
def _table_embed_text(name: str, cols: list[str]) -> str:
    """Text embedded per table: its name plus column names — the discriminating
    signal we have (introspected tables carry no description)."""
    if cols:
        return f"{name} — columns: {', '.join(cols)}"
    return name


def embed_tables(driver: Driver, batch: int = 64) -> None:
    """Embed each Table from its name + column names into Table.embedding and ensure
    the `table_embeddings` vector index exists.

    Always real (unlike embed_chunks it ignores fake_embeddings): schema routing
    retrieves over these vectors, so they must carry real semantics."""
    client = get_openai_client()
    with driver.session(database=settings.neo4j_database) as session:
        rows = session.run(
            """
            MATCH (t:Table)
            OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
            WITH t, collect(c.name) AS cols
            RETURN t.id AS id, t.name AS name, cols
            """
        ).data()
        for i in range(0, len(rows), batch):
            window = rows[i:i + batch]
            texts = [_table_embed_text(r["name"], r["cols"]) for r in window]
            vectors = client.embeddings.create(
                model=settings.embedding_model, input=texts,
                dimensions=settings.embedding_dimensions,
            ).data
            session.run(
                """
                UNWIND $rows AS row
                MATCH (t:Table {id: row.id})
                CALL db.create.setNodeVectorProperty(t, 'embedding', row.vec)
                """,
                rows=[{"id": w["id"], "vec": v.embedding} for w, v in zip(window, vectors)],
            )
    _ensure_table_vector_index(driver)


def _ensure_table_vector_index(driver: Driver) -> None:
    """Create a vector index named `table_embeddings` on Table.embedding — a stable
    name we own (mirroring `chunk_embeddings`). Drop any NeoCarta-named index on the
    same property first to avoid a duplicate-index error."""
    with driver.session(database=settings.neo4j_database) as session:
        session.run("DROP INDEX table_vector_index IF EXISTS")
        session.run(
            f"""
            CREATE VECTOR INDEX table_embeddings IF NOT EXISTS
            FOR (t:Table) ON (t.embedding)
            OPTIONS {{indexConfig: {{
              `vector.dimensions`: {settings.embedding_dimensions},
              `vector.similarity_function`: 'cosine'
            }}}}
            """
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_embed_tables.py -v`
Expected: the two `_table_embed_text` tests PASS. The marked
`test_embed_tables_writes_embeddings_and_index` PASSES if Neo4j is up and
`OPENAI_API_KEY` is set (use `backend/.venv/bin/python`); otherwise it SKIPs. Run the
marked test when a key is available to get real evidence:
`OPENAI_API_KEY=$OPENAI_API_KEY .venv/bin/python -m pytest tests/test_embed_tables.py -v -m "neo4j and openai"`.

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/ingest/embeddings.py backend/tests/test_embed_tables.py
git commit -m "feat(routing): own table embeddings + table_embeddings vector index"
```

---

### Task 4: Hybrid `retrieve_candidate_tables`

**Files:**
- Modify: `backend/semantic_layer/agent/routing.py` (rewrite `retrieve_candidate_tables`, add helpers + imports)
- Test: `backend/tests/test_routing.py` (update one test, add two)

**Interfaces:**
- Consumes: `embed_query` (Task 2), `settings.schema_routing_k_vec` (Task 1), existing `search_catalog`, `driver`.
- Produces: `_vector_table_hits(question, k) -> dict[str, float]`; `_keyword_value_hits(question) -> dict[str, float]`; `_keyword_fallback(question, k_ret) -> list[dict]`; rewritten `retrieve_candidate_tables(question, k_ret=20) -> list[dict]` (same shape).

- [ ] **Step 1: Update + add tests**

In `backend/tests/test_routing.py`, REPLACE the existing
`test_retrieve_candidate_tables_dedups_and_ranks` (lines 7-20) with a fallback test, and
add two hybrid tests after it:

```python
def test_retrieve_falls_back_to_keyword_when_no_vector(monkeypatch):
    # No table embeddings -> vector hits empty -> keyword fallback (prior behavior).
    monkeypatch.setattr(routing, "_vector_table_hits", lambda q, k: {})
    hits = [
        {"kind": "value", "id": "c1", "name": "EMEA", "table_id": "table:sales_pg.sales.region", "score": 3},
        {"kind": "column", "id": "c2", "name": "segment", "table_id": "table:sales_pg.sales.segment", "score": 2},
        {"kind": "value", "id": "c3", "name": "EMEA", "table_id": "table:sales_pg.sales.region", "score": 1},
    ]
    monkeypatch.setattr(routing, "search_catalog", type("T", (), {
        "invoke": staticmethod(lambda _a: json.dumps(hits))})())
    out = routing.retrieve_candidate_tables("revenue by segment in EMEA", k_ret=20)
    ids = [c["table_id"] for c in out]
    assert ids == ["table:sales_pg.sales.region", "table:sales_pg.sales.segment"]
    assert out[0]["score"] == 4  # 3 + 1 for region


def test_retrieve_unions_vector_and_value_hits(monkeypatch):
    monkeypatch.setattr(routing, "_vector_table_hits",
                        lambda q, k: {"table:sales_pg.sales.customer": 0.9})
    # search_catalog supplies a value hit for a different table (region via 'EMEA').
    hits = [{"kind": "value", "id": "c1", "name": "EMEA",
             "table_id": "table:sales_pg.sales.region", "score": 3}]
    monkeypatch.setattr(routing, "search_catalog", type("T", (), {
        "invoke": staticmethod(lambda _a: json.dumps(hits))})())
    out = routing.retrieve_candidate_tables("customers in EMEA", k_ret=20)
    ids = {c["table_id"] for c in out}
    assert "table:sales_pg.sales.customer" in ids   # from vector
    assert "table:sales_pg.sales.region" in ids     # from value keyword hit


def test_retrieve_keyword_value_hits_ignores_non_value_kinds(monkeypatch):
    monkeypatch.setattr(routing, "_vector_table_hits",
                        lambda q, k: {"table:sales_pg.sales.customer": 0.9})
    # a plain column hit must NOT be unioned (only value/business_term routing is kept)
    hits = [{"kind": "column", "id": "c2", "name": "amount",
             "table_id": "table:scale.scale_hr.payroll", "score": 5}]
    monkeypatch.setattr(routing, "search_catalog", type("T", (), {
        "invoke": staticmethod(lambda _a: json.dumps(hits))})())
    out = routing.retrieve_candidate_tables("how many customers", k_ret=20)
    ids = {c["table_id"] for c in out}
    assert ids == {"table:sales_pg.sales.customer"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_routing.py -v`
Expected: the new tests FAIL with `AttributeError: ... has no attribute '_vector_table_hits'`.

- [ ] **Step 3: Rewrite `retrieve_candidate_tables` in `routing.py`**

In `backend/semantic_layer/agent/routing.py`, update the imports block (lines 6-13) to add:

```python
from neo4j.exceptions import ClientError

from semantic_layer.ingest.embeddings import embed_query
```

Then REPLACE the whole `retrieve_candidate_tables` function (lines 33-48) with:

```python
_VALUE_KINDS = {"value", "business_term"}


def _vector_table_hits(question: str, k: int) -> dict[str, float]:
    """Top-k tables by cosine similarity of the question to Table.embedding.
    {table_id: score}. Empty when the `table_embeddings` index is missing/unbuilt,
    so the caller falls back to keyword retrieval."""
    try:
        vec = embed_query(question)
        recs = driver().execute_query(
            """
            CALL db.index.vector.queryNodes('table_embeddings', $k, $vec)
            YIELD node, score
            RETURN node.id AS table_id, score
            """,
            k=k, vec=vec, database_=settings.neo4j_database,
        ).records
    except ClientError:
        return {}
    return {r["table_id"]: r["score"] for r in recs
            if r["table_id"] and r["table_id"].startswith("table:")}


def _keyword_value_hits(question: str) -> dict[str, float]:
    """Exact value / business-term routing from search_catalog (e.g. 'EMEA' -> region
    table) — the signal embeddings cannot provide. Only value/term hits resolving to a
    real table id are kept."""
    hits = json.loads(search_catalog.invoke({"query": question}))
    out: dict[str, float] = {}
    for h in hits:
        if h.get("kind") not in _VALUE_KINDS:
            continue
        tid = h.get("table_id")
        if not tid or not tid.startswith("table:"):
            continue
        out[tid] = out.get(tid, 0.0) + float(h.get("score") or 1)
    return out


def _keyword_fallback(question: str, k_ret: int) -> list[dict]:
    """Original keyword-only aggregation over all search_catalog hits, used when no
    table embeddings exist so retrieval degrades to prior behavior."""
    hits = json.loads(search_catalog.invoke({"query": question}))
    scores: dict[str, float] = {}
    for h in hits:
        tid = h.get("table_id")
        if not tid or not tid.startswith("table:"):
            continue
        scores[tid] = scores.get(tid, 0.0) + float(h.get("score") or 1)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"table_id": tid, "score": s} for tid, s in ranked[:k_ret]]


def retrieve_candidate_tables(question: str, k_ret: int = 20) -> list[dict]:
    """High-recall candidate tables: semantic vector hits over Table.embedding unioned
    with exact value-routing keyword hits. Falls back to keyword-only when no table
    embeddings exist. Returns [{table_id, score}] capped at k_ret; the LLM ranker
    (rank_tables) trims to a precise set."""
    vector = _vector_table_hits(question, k=settings.schema_routing_k_vec)
    if not vector:
        return _keyword_fallback(question, k_ret)
    scores: dict[str, float] = dict(vector)
    for tid in _keyword_value_hits(question):
        scores.setdefault(tid, 1.0)   # ensure value-matched tables make the candidate set
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"table_id": tid, "score": s} for tid, s in ranked[:k_ret]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_routing.py -v`
Expected: PASS. The three updated/new unit tests pass; the pre-existing
`test_retrieve_candidate_tables_finds_real_dimensions` (neo4j) still passes because the
ingested fixture has no table embeddings, so `_vector_table_hits` hits the missing index,
returns `{}`, and the keyword fallback preserves the old behavior (`rank_tables`/`route_tables`
tests unaffected). If Neo4j is down those marked tests SKIP.

- [ ] **Step 5: Add an end-to-end vector-routing test**

Append to `backend/tests/test_routing.py`:

```python
@pytest.mark.neo4j
@pytest.mark.openai
def test_vector_routing_finds_customer_table(ingested_graph):
    # Embed tables for real, then the semantic query that keyword retrieval missed.
    from semantic_layer.ingest.embeddings import embed_tables
    embed_tables(ingested_graph)
    out = routing.retrieve_candidate_tables("How many customers are there in total?", k_ret=20)
    ids = {c["table_id"] for c in out}
    assert "table:sales_pg.sales.customer" in ids
```

- [ ] **Step 6: Run the e2e test (if key available)**

Run: `cd backend && OPENAI_API_KEY=$OPENAI_API_KEY .venv/bin/python -m pytest tests/test_routing.py::test_vector_routing_finds_customer_table -v`
Expected: PASS (the `customer` table is retrieved via vector similarity). SKIPs without
Neo4j/key.

- [ ] **Step 7: Commit**

```bash
git add backend/semantic_layer/agent/routing.py backend/tests/test_routing.py
git commit -m "feat(routing): hybrid vector+keyword retrieve_candidate_tables with fallback"
```

---

### Task 5: Wire `embed_tables` into ingest; remove dead `embed_metadata_nodes`

**Files:**
- Modify: `backend/semantic_layer/ingest/pipeline.py:109-132` (`_run_llm_stages` import + call)
- Modify: `backend/semantic_layer/ingest/embeddings.py` (remove `embed_metadata_nodes` + its NeoCarta import)
- Test: `backend/tests/test_pipeline_embeds_tables.py`

**Interfaces:**
- Consumes: `embed_tables` (Task 3).
- Produces: `_run_llm_stages` now calls `embed_tables(driver)` instead of `embed_metadata_nodes(driver)`; `embed_metadata_nodes` no longer exists.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_pipeline_embeds_tables.py
import inspect

from semantic_layer.ingest import pipeline, embeddings


def test_llm_stages_embeds_tables_not_metadata_nodes():
    src = inspect.getsource(pipeline._run_llm_stages)
    assert "embed_tables(" in src
    assert "embed_metadata_nodes" not in src


def test_embed_metadata_nodes_removed():
    assert not hasattr(embeddings, "embed_metadata_nodes")
    assert hasattr(embeddings, "embed_tables")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_pipeline_embeds_tables.py -v`
Expected: FAIL (`embed_metadata_nodes` still present in source / module).

- [ ] **Step 3: Swap the call in `pipeline.py`**

In `backend/semantic_layer/ingest/pipeline.py`, inside `_run_llm_stages`, change the
import line (currently
`from semantic_layer.ingest.embeddings import embed_chunks, embed_metadata_nodes`) to:

```python
    from semantic_layer.ingest.embeddings import embed_chunks, embed_tables
```

and change the two trailing calls (currently `embed_chunks(driver)` then
`embed_metadata_nodes(driver)`) to:

```python
    embed_chunks(driver)
    embed_tables(driver)
```

- [ ] **Step 4: Remove the dead function in `embeddings.py`**

In `backend/semantic_layer/ingest/embeddings.py`, delete the entire
`embed_metadata_nodes` function (lines 94-106) and remove the now-unused import line
`from neocarta.enrichment.embeddings import OpenAIEmbeddingsConnector` (line 8).

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_pipeline_embeds_tables.py tests/test_embed_tables.py tests/test_pipeline_scale.py -v`
Expected: PASS (wiring asserted; embed_tables tests still pass; scale pipeline tests
unaffected).

- [ ] **Step 6: Commit**

```bash
git add backend/semantic_layer/ingest/pipeline.py backend/semantic_layer/ingest/embeddings.py backend/tests/test_pipeline_embeds_tables.py
git commit -m "feat(routing): ingest embeds tables (drops no-op NeoCarta metadata embed)"
```

---

### Task 6: Validation — live re-ingest + re-eval (controller-run, not a subagent task)

This task is **not** a code change and **not** for an implementer subagent — the
controller runs it after the code tasks are merged, because it rebuilds the live Neo4j
graph and calls OpenAI. The dev environment is already in scale state (5k/50k core + 1,000
distractor tables seeded), so only a re-ingest is needed (no re-seed).

- [ ] **Step 1: Re-ingest the scaled graph (now embeds tables for real)**

```bash
cd backend && SCALE_MODE=true SCHEMA_ROUTING_ENABLED=true FAKE_EMBEDDINGS=true \
  .venv/bin/python -m semantic_layer.ingest.pipeline
```

`embed_tables` ignores `FAKE_EMBEDDINGS`, so tables are embedded for real even with the
flag on (chunks stay fake — fine, doc-RAG isn't under test). Expect counts including
`scale_sources` and no error.

- [ ] **Step 2: Confirm the table_embeddings index is populated**

```bash
cd backend && .venv/bin/python -c "
from semantic_layer.graph.client import get_driver
from semantic_layer.config import settings
d=get_driver()
with d.session(database=settings.neo4j_database) as s:
    miss=s.run(\"MATCH (t:Table) WHERE t.embedding IS NULL RETURN count(t) AS c\").single()['c']
    print('tables missing embedding:', miss)
d.close()"
```

Expected: `tables missing embedding: 0`.

- [ ] **Step 3: Re-run the routing eval and compare to baseline**

```bash
cd backend && SCHEMA_ROUTING_ENABLED=true \
  .venv/bin/python -m eval.run_eval --routing-only --out scorecard-routing-v2.json
```

Expected: a printed scorecard. **Success criterion:** routing hit-rate and recall are
materially higher than the 17.4% / 0.26 baseline (`scorecard-routing.json`). If not,
report the numbers and the worst-recall questions for a follow-up (e.g. raise
`SCHEMA_ROUTING_K_VEC`, enrich the embed text).

- [ ] **Step 4: Record the result**

Note the before/after numbers in the branch summary / ledger. No commit (the scorecards
are run artifacts).

---

## Notes for the implementer

- **Routing previously did NOT use embeddings** — this plan is the first vector path for
  tables. The `table_embeddings` index name is owned by us (like `chunk_embeddings`), not
  NeoCarta's auto-generated `table_vector_index`.
- **Fallback is load-bearing:** any graph ingested before Task 5 (no table embeddings)
  must keep routing — that's why `_vector_table_hits` returns `{}` on a missing index and
  `retrieve_candidate_tables` falls back to the full keyword aggregation.
- **Do not run the full pytest suite** — per project policy it wipes document embeddings.
  Run the focused files named in each task. After Task 6's live re-ingest, doc embeddings
  are rebuilt (chunks fake under `FAKE_EMBEDDINGS=true`); run `make ingest` (real) if you
  need real doc-RAG quality back.
- **OpenAI key:** the marked (`openai`) tests and Task 6 need `OPENAI_API_KEY` (it is in
  `backend/.env` / `settings.openai_api_key`); use `backend/.venv/bin/python` so the
  dependency set and key are available.
