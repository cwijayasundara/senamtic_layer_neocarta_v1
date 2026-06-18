# NeoCarta-Local — Unified Semantic + Context Layer over Databases, APIs, and Documents

**Date:** 2026-06-16
**Status:** Approved design, pending implementation plan

## 1. Purpose & Goal

Build a locally-runnable sample application that recreates the **NeoCarta semantic-layer pattern** (Neo4j Labs) so that AI agents can query heterogeneous enterprise data **uniformly** — without knowing which physical source holds an answer. The system demonstrates a semantic/context layer spanning three source types:

1. **Structured databases** (SQLite + PostgreSQL)
2. **Enterprise APIs** (mocked REST services)
3. **Unstructured documents** (NVIDIA press-release PDFs)

All data is NVIDIA-themed and synthetic (except the two real PDFs already in `docs/`).

### Success criteria
- A single chat interface answers questions that route to the correct source(s) automatically.
- A structured question requiring a **6+-table join** is answered correctly, with the join path derived from the graph.
- An API-only question and a **cross-source** question (DB + API + document) are each answered with citations/provenance.
- The UI **visualizes the semantic graph and animates the traversal path** the agent took.
- The whole stack runs locally via `docker-compose` + documented commands.

## 2. Background: NeoCarta vs. create-context-graph

| | NeoCarta (`neo4j-labs/neocarta`) | create-context-graph (`neo4j-labs/create-context-graph`) |
|---|---|---|
| Type | Python **library** (ETL: Extractors→Transformers→Loaders→Connectors) | Interactive **CLI scaffolder** that generates a full-stack app |
| Builds | *Metadata + semantic* graph: `Database/Schema/Table/Column/Value/BusinessTerm/Glossary/Category` + `REFERENCES`, **`Query`/`CTE` from mined query logs**, and **OSI semantic models** (`OsiSemanticModel/Metric/Join/Expression`) | *Entity/context* graph (POLE+O) from unstructured + SaaS data, with provenance/memory |
| For | Query routing, **Text2SQL**, join discovery, data discovery over **structured DBs**, **query-log usage analysis**, and **semantic-spec (OSI) ingestion** | Conversational/entity knowledge over **unstructured + SaaS** sources |
| Exposes | MCP server with vector/full-text/**hybrid** search | FastAPI + Next.js app, 8 agent frameworks, 13+ connectors |

> **Capability note (verified against `neocarta==0.7.0`, 2026-06-18).** NeoCarta is broader than "structured DBs only": shipped connectors are `bigquery` (schema + **logs**), `csv`, `dataplex` (schema + glossary), `osi`, and `query_log`; node labels include `Query`, `CTE`, and the full OSI `Metric/Join/Domain` family (`neocarta/enums.py`). What NeoCarta does **not** have is an **unstructured-document** connector — there is no PDF/text/chunk/entity pipeline and `enrichment/embeddings` embeds *metadata* node descriptions (`[TABLE, COLUMN]`), not prose. So our reimplementation of the document/context layer is still required, but NeoCarta's **query-log mining** and **OSI semantic models** are real capabilities we can adopt rather than rebuild. (A vendor diagram showing "documents, ontologies, query logs" refers to structured/semantic-spec inputs — OSI/glossary — not unstructured-document RAG.)

**Decision:** Hybrid. Use the **NeoCarta library** as the metadata-semantic-layer engine for structured DBs *and* the mocked APIs (a custom Extractor models each API's OpenAPI spec as virtual tables/columns), **and mine query logs via NeoCarta's `query_log` connector to enrich join-path discovery** (see §5). **Reimplement create-context-graph's entity + provenance idea lightly** for documents (NeoCarta has no unstructured-document connector). Keep our **own deepagents orchestration and a custom Next.js UI** rather than create-context-graph's stock Chakra app. The two graph halves join via shared `BusinessTerm`/`Entity` nodes — that bridge *is* the semantic layer. **OSI semantic models remain an unused-but-available path** for curated metrics, should we want them over hand-rolled `BusinessTerm` generation.

## 3. Data Sources (synthetic, NVIDIA-themed)

### 3.1 Structured databases
- **PostgreSQL (Docker) — `sales` schema, 11 normalized tables** designed so realistic questions require deep joins:
  `region → country → customer → industry`, `sales_order → fiscal_period`, `order_line → product → product_line → segment` + `architecture`.
  Tables: `region`, `country`, `customer`, `industry`, `sales_order`, `order_line`, `product`, `product_line`, `segment`, `architecture`, `fiscal_period`.
  Example deep-join question: *"Total Data Center revenue from Cloud customers in EMEA for Blackwell products in Q4 FY2025"* → joins `order_line · product · product_line · segment · architecture · sales_order · fiscal_period · customer · industry · country · region` (8–11 tables).
- **SQLite `financials.db`:** quarterly income statement (revenue, gross margin, net income, EPS) + daily stock prices.
- **SQLite `org.db`:** headcount by department, locations, hiring trend.

### 3.2 Mock enterprise APIs (standalone FastAPI services, each with an OpenAPI spec + synthetic data)
- **CRM (Salesforce-like):** accounts, contacts, opportunities, pipeline stages.
- **Support / ITSM (ServiceNow-like):** tickets, severities, SLAs, GPU RMA.
- **Partner / channel inventory:** distributor stock levels and allocations per product line.
- **DGX Cloud usage / telemetry:** per-customer GPU-hours, utilization, instance types (time-series).

### 3.3 Documents
- The two existing PDFs in `docs/` (`NVIDIAAn_2025.pdf`, `NVIDIAAn_2026.pdf`), parsed with **liteparse v2**.

## 4. The Semantic + Context Graph (Neo4j)

Three layers in one Neo4j database:

1. **Metadata layer (NeoCarta):** `Source → Database → Schema → Table → Column`, with `REFERENCES` edges from foreign keys (declared + naming heuristics), plus LLM-generated `BusinessTerm` nodes (`TAGGED_WITH` columns/tables). A **custom NeoCarta Extractor** maps each mock API's OpenAPI spec into the same shape: `Endpoint` ≈ Table, `Field/Param` ≈ Column, with the API as a `Source`.
2. **Context/entity layer (create-context-graph pattern, reimplemented):** `Document → Chunk` (embedded), POLE+O `Entity` nodes (Person/Org/Location/Event/Object) extracted from documents and key DB/API values, `MENTIONS` edges, with **provenance** (which source/passage produced each fact).
3. **Bridge + search indexes:** shared `BusinessTerm`/`Entity` nodes link DB columns ↔ API fields ↔ document chunks (e.g. "Blackwell" links `architecture.name='Blackwell'`, a CRM opportunity field, and press-release chunks). Neo4j **vector + full-text indexes** support NeoCarta-style **hybrid search**.

## 5. Ingestion (build-time pipeline)

- **NeoCarta SQL extractors:** introspect Postgres + both SQLite DBs → metadata-layer nodes + `REFERENCES` (from *declared* FKs).
- **Custom API extractor:** read each mock API's OpenAPI spec → virtual `Endpoint`/`Field` nodes.
- **NeoCarta query-log connector (`query_log_indexer.py`):** parse a BigQuery-audit-log-shaped query log (`data/seed/query_log_sales.json`) with NeoCarta's `QueryLogExtractor`, then bridge its discovered joins onto our canonical catalog as weighted `(:Column)-[:OBSERVED_JOIN {observations}]->(:Column)` edges (+ a `:Query` provenance node). Because NeoCarta's parser emits dotted ids (`project.dataset.table`) while we use prefixed ids (`table:source.schema.table`), we resolve each join column by **(table name, column name) across all SQL sources** rather than load NeoCarta's structural duplicates — the same extract-then-bridge pattern used for APIs and documents. Ambiguous name pairs (same table+column name in two sources) are skipped, not guessed. Two payoffs, both covered by `get_join_path`:
  - **Discovery of FK-less joins.** `get_join_path` traverses `OBSERVED_JOIN` alongside `REFERENCES`/`SAME_ENTITY`, so joins **no FK declares** become first-class. The shipped log includes a cross-database join — `financials.income_statement ⋈ org.headcount` on `fiscal_year` (revenue vs headcount by year) — which has **zero** REFERENCES edges anywhere yet is fully plannable via its `OBSERVED_JOIN` (see `test_query_log_indexer.py`).
  - **Usage ranking.** `get_join_path` uses `allShortestPaths` and ranks equally-short paths by **total `OBSERVED_JOIN` weight**, so an empirically-travelled join wins over an FK-only guess of equal length.
  Optional and non-fatal: ingest skips it when the log file is absent.
- **Document pipeline:** `liteparse` (`LiteParse().parse(path).text` / `.pages`) → chunk → embed; LLM entity extraction (POLE+O) → entity layer + `MENTIONS` + provenance.
- **LLM glossary:** `gpt-5.4-mini` generates `BusinessTerm` nodes + descriptions from schema; embedded.
- **Embeddings:** OpenAI `text-embedding-3-small`.
- **Indexes:** create vector + full-text indexes after load.
- **Idempotent:** all writes via Cypher `MERGE`; pipeline is re-runnable. Per-document parse failures are logged, not fatal.

## 6. Agent Layer (deepagents)

All models created via **`init_chat_model("openai:gpt-5.4-mini")`** and passed to `create_deep_agent(...)`.

- **Orchestrator deep agent:** built-in `write_todos` planning + virtual filesystem for context offload. Tools (semantic-layer routers): `search_catalog(q)` (hybrid vector+full-text), `get_table_schema(table)`, **`get_join_path(table_a, table_b)`** (shortest path over `REFERENCES` edges — the concrete payoff of the graph for deep joins), `list_sources()`.
- **Subagents (context isolation via the `task` tool):**
  - **`sql_subagent`** — grounded Text2SQL using LangChain `SQLDatabase` + the four SQL tools (`sql_db_list_tables`, `sql_db_schema`, `sql_db_query_checker`, `sql_db_query`). Receives the orchestrator's chosen tables + `get_join_path` result so it is *not* doing blind text-to-SQL. Read-only (SELECT-only, no DML), row limit, query timeout, one self-repair attempt on error.
  - **`api_subagent`** — generates and executes REST calls against the mock APIs, grounded by the API metadata (endpoint/param schema retrieved from the graph).
  - **`doc_subagent`** — document RAG via `search_documents(q)` over chunk embeddings; returns passages + citations.
- **Flow:** plan → semantic retrieval → route (sql / api / doc / combination) → execute subagent(s) → synthesize answer with citations + provenance trace.

### Why text-to-SQL is still required
The semantic layer stores **metadata, not relational rows**. Document *content* lives in the graph (→ direct RAG), but structured answers (revenue figures, GPU-hours, ticket counts) must be fetched live from Postgres/SQLite (SQL) or the APIs (REST). The graph's job is to tell the agent *which source, tables, fields, and join path* — then the subagent fetches real values.

## 7. Backend (FastAPI)

- `POST /chat` → **SSE stream** of events: plan steps (todos), route decision, generated SQL / REST call, graph traversal path, answer tokens, citations/provenance.
- `GET /graph` → graph nodes/edges for visualization; supports a per-query subgraph for the highlight.
- `GET /sources` → source catalog.
- CORS enabled for the Next.js dev server. Mock API services run alongside (separate ports or mounted sub-apps).
- *Optional / out of scope for v1:* re-expose NeoCarta's MCP server.

## 8. Frontend (Next.js + React)

- **Stack:** Next.js (App Router), TypeScript, Tailwind + shadcn/ui, a graph renderer (`react-force-graph` / Cytoscape), SSE consumption.
- **Split-canvas layout:** chat on the left, live semantic-layer visualization on the right.
- **Signature feature — animated graph traversal:** on each answer, the right canvas renders the Neo4j graph and lights up the path the agent took — `BusinessTerm`/`Entity` → resolved tables/endpoints → `get_join_path` chain → source. Document answers highlight `Document → Chunk → Entity`.
- **Reasoning-trace panel:** streams the deepagents `write_todos` plan as it forms, the route decision, the generated SQL (syntax-highlighted, join path annotated) or REST call, and doc citations as cards linking to the PDF page/passage.
- **Token streaming** for live answers.

## 9. Operations & Repo Layout

```
sementic_layer_neocarta_v1/
  docker-compose.yml          # neo4j + postgres
  Makefile                    # up · seed · serve-apis · ingest · run
  .env.example                # OPENAI_API_KEY, NEO4J_*, POSTGRES_*
  docs/                       # the 2 PDFs (+ this spec under superpowers/specs/)
  backend/
    pyproject.toml
    semantic_layer/
      config.py
      sources/                # postgres + sqlite connectors, source registry
      apis/                   # 4 mock enterprise API services (FastAPI) + OpenAPI specs + synthetic data
      ingest/
        sql_extractor.py      # NeoCarta-based SQL metadata extractor
        api_extractor.py      # custom NeoCarta extractor: OpenAPI -> virtual tables/columns
        doc_parser.py         # liteparse v2
        entities.py           # POLE+O entity extraction + provenance
        glossary.py           # LLM-generated BusinessTerms
        embeddings.py
        graph_builder.py      # MERGE schema/entity/bridge layers + indexes
      graph/                  # neo4j driver client + Cypher (queries, join-path, indexes)
      tools/                  # search_catalog, get_table_schema, get_join_path, list_sources, search_documents
      agent/                  # deepagents orchestrator + sql/api/doc subagents + prompts
      api/                    # FastAPI app: /chat (SSE), /graph, /sources
    data/generators/          # synthetic NVIDIA data scripts
    tests/
  frontend/                   # Next.js app (chat + animated graph traversal + trace)
```

## 10. Error Handling

- SQL: read-only guard (reject non-SELECT/DML), row limit, timeout, one LLM self-repair on execution error.
- API calls: validated against the endpoint schema before execution; network/4xx-5xx errors surfaced to the agent for one retry.
- Ingestion: idempotent `MERGE`; per-document parse failures logged, not fatal.
- Startup health checks for Neo4j, Postgres, mock APIs, and `OPENAI_API_KEY`.
- Connection retries on startup for Neo4j/Postgres.

## 11. Testing

- **Unit:** SQL read-only guard; `get_join_path` graph traversal; NeoCarta API extractor (OpenAPI spec → graph nodes); liteparse parsing of a small fixture; entity extraction; retrieval ranking.
- **Integration golden questions** (seed → ingest → ask, asserting correct source routing and answer):
  - (a) a **6+-table-join** structured question;
  - (b) an **API-only** question;
  - (c) a **cross-source** question spanning DB + API + document.

## 12. Build Phases (detailed steps come in the implementation plan)

1. Scaffold (`backend/` + `frontend/`) + `docker-compose` + config/health checks.
2. DB synthetic data — normalized 11-table Postgres `sales` + 2× SQLite — + seed.
3. Mock enterprise APIs (CRM, ITSM, partner inventory, DGX Cloud) with OpenAPI specs + synthetic data.
4. liteparse v2 document parsing + chunking.
5. NeoCarta metadata layer — SQL extractors + custom API/OpenAPI extractor → graph (`REFERENCES`, endpoints).
6. Context/entity layer (POLE+O) + LLM glossary `BusinessTerm`s + embeddings + hybrid indexes + bridge links.
7. Semantic tools (hybrid `search_catalog`, `get_table_schema`, `get_join_path`, `list_sources`, `search_documents`).
8. deepagents orchestrator + `sql_subagent` / `api_subagent` / `doc_subagent`.
9. FastAPI — `/chat` SSE, `/graph`, `/sources`.
10. Next.js innovative UI — split canvas, animated traversal, reasoning trace.
11. Tests + golden questions + README.

## 13. Out of Scope (v1, YAGNI)

- Re-exposing a NeoCarta MCP server (tools are deepagents tools instead).
- Real (non-mock) external APIs or live SaaS connectors.
- Authentication / multi-tenant concerns.
- Production deployment (local-only).

## 14. Caching (production scaling, P0)

A query cache sits in front of `answer_stream` so repeated or near-identical
questions skip re-running intent extraction, legs, and synthesis. See the plan
`docs/superpowers/plans/2026-06-18-production-scaling-p0.md` (Feature C).

- **Implemented (default):** an in-process `QueryCache` (`semantic_layer/agent/cache.py`)
  with exact-match (normalized question) plus semantic (embedding-cosine) lookup,
  LRU-bounded with TTL. Gated on `settings.query_cache_enabled`; tunables
  `cache_max_entries`, `cache_ttl_seconds`, `cache_similarity_threshold`.
- **Single-worker only.** The in-process cache is **per-process**: it is not
  shared across uvicorn workers or replicas. Two workers keep independent caches,
  so a hit rate degrades under horizontal scaling and entries are never
  cross-invalidated.
- **Production path — `RedisQueryCache` (follow-up, not yet built).** For
  multi-worker / multi-replica deploys, add a Redis-backed implementation of the
  same `get_exact` / `get_semantic` / `put` interface, keyed by normalized
  question. Semantic lookup should be backed by a **Redis vector index
  (RediSearch)** rather than the in-process linear scan, so similarity search
  stays sub-linear as the cache grows.
- **Selection.** Gate the backend behind a `cache_backend: str = "memory" | "redis"`
  setting plus a `redis_url`; default `"memory"`. Add `redis>=5` as an optional
  `pyproject.toml` extra only when the Redis backend is implemented (YAGNI until a
  multi-worker deploy is real).
