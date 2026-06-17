# NeoCarta-Local — Backend (Plan 1: Data Foundation)

## Prerequisites
- Docker + Docker Compose
- Python 3.11+

## Setup
```bash
cp backend/.env.example backend/.env        # adjust if needed
make up                                      # start postgres + neo4j

python3 -m venv backend/.venv                # create the virtualenv
source backend/.venv/bin/activate            # activate it (Makefile targets need this active)
make install                                 # pip install -e ".[dev]"

make seed                                    # load postgres sales schema + sqlite databases
make test                                    # run the full test suite
```

> The Makefile's `test`, `seed`, and `install` targets call bare `python`/`pip`,
> so the `backend/.venv` virtualenv must be **activated** first (the `source ...`
> line above). Without it those targets use system Python and will fail to import deps.

## What this provides
- Postgres `sales` schema: 11 normalized tables supporting 6+-table joins.
- SQLite `financials.db` (income statement + stock prices) and `org.db` (headcount).
- Deterministic, reproducible synthetic NVIDIA-themed data (seeded RNG).

## Verify the deep join
After `make seed`, the 11-table join in `tests/test_seed_postgres.py`
(`test_deep_eleven_table_join_returns_rows`) confirms a question like
"Data Center revenue for Blackwell products" resolves across
order_line → product → product_line → segment/architecture →
sales_order → fiscal_period → customer → industry → country → region.

## Next plans
2. Mock enterprise APIs · 3. Graph ingestion · 4. deepagents agent · 5. Web app.

## Mock enterprise APIs (Plan 2)

Four NVIDIA-themed mock REST APIs run as one Uvicorn process (mounted sub-apps),
serving deterministic in-memory data. Accounts reuse the sales customer base, so
API records join back to the `sales` data by `account_id` (== `customer_id`).

```bash
source backend/.venv/bin/activate
make serve-apis     # uvicorn on http://localhost:8001
```

| API | Prefix | Key endpoints | OpenAPI |
|-----|--------|---------------|---------|
| CRM | `/crm` | `/accounts`, `/contacts`, `/opportunities` | `/crm/openapi.json` |
| Support/ITSM | `/itsm` | `/tickets`, `/rma` | `/itsm/openapi.json` |
| Partner inventory | `/partner` | `/partners`, `/inventory` | `/partner/openapi.json` |
| DGX Cloud telemetry | `/dgx` | `/usage` | `/dgx/openapi.json` |

Health check: `GET /health`. These OpenAPI specs are what Plan 3's NeoCarta API
extractor introspects into the semantic graph as virtual tables/columns.

## Graph ingestion (Plan 3)

Builds the unified semantic + context graph in Neo4j from all three source types.

Prerequisites: `make up` (Neo4j + Postgres), `make seed` (databases), and
`OPENAI_API_KEY` in `backend/.env` (for entities, glossary, embeddings).

```bash
source backend/.venv/bin/activate
make ingest        # python -m semantic_layer.ingest.pipeline
```

The pipeline runs idempotently (resets, then MERGEs) in this order:

1. **Metadata layer (NeoCarta):** Postgres + 2 SQLite DBs + 4 mock-API OpenAPI
   specs -> `Database -> Schema -> Table -> Column` with `REFERENCES` (FK) edges.
   APIs are modelled as virtual tables (endpoints) / columns (response fields).
2. **Document layer:** the NVIDIA PDFs are parsed with liteparse v2, chunked,
   and stored as `Document -> Chunk` nodes.
3. **Entity layer:** POLE+O entities extracted from chunks via the LLM, linked
   `Chunk -> MENTIONS -> Entity` (provenance = the chunk).
4. **Glossary bridge:** LLM-generated `BusinessTerm` nodes tagged onto columns
   via `Column -> TAGGED_WITH -> BusinessTerm`.
5. **Embeddings:** chunk + metadata-node embeddings (`text-embedding-3-small`,
   1536-d) with vector indexes for hybrid search.

Re-running `make ingest` rebuilds the graph from scratch (idempotent).

## Ask the agent (Plan 4)

A deepagents orchestrator (on `gpt-5.4-mini`) answers natural-language questions
across all sources. It uses graph-backed tools to route — `search_catalog`,
`get_table_schema`, `get_join_path` (deep-join discovery), `search_documents` —
and delegates to three subagents: **sql** (grounded read-only text-to-SQL over
Postgres/SQLite), **api** (the CRM/ITSM/partner/DGX mock APIs), and **doc**
(vector RAG over the document chunks). It then synthesizes an answer with provenance.

Prerequisites: `make up`, `make seed`, `make ingest`, and `OPENAI_API_KEY` in `backend/.env`.

```bash
source backend/.venv/bin/activate
make ask q="Which business segment has the highest total revenue?"
make ask q="How many open support tickets are there?"
make ask q="According to the press releases, what drove Data Center growth?"
make ask q="List EMEA Cloud customers and their open support tickets."   # cross-source
make ask q="In FY2025, which EMEA Cloud customers bought Blackwell Data Center products, and what was each customer's total revenue by quarter?"   # 11-table deep join
make ask q="Compare the Data Center revenue we recorded for Blackwell products with what the NVIDIA press releases say drove Data Center growth."   # SQL + document RAG
```

The 11-table question forces the orchestrator to traverse the full `sales`
schema via `get_join_path`: `region`(EMEA) → `country` → `customer` →
`industry`(Cloud), joined to `segment`(Data Center) ← `product_line` →
`architecture`(Blackwell), then `product` → `order_line`(revenue) ←
`sales_order` → `fiscal_period`(FY2025, quarter). The comparison question fans
out to **two** subagents at once — the **sql** subagent runs the Data
Center/Blackwell revenue join while the **doc** subagent vector-searches the
NVIDIA press-release PDFs — and the orchestrator reconciles the recorded
numbers against the documents' growth narrative in a single provenance-tagged answer.

The sql subagent is **grounded**: the orchestrator hands it the resolved tables,
their `sql_reference`, and the `get_join_path` chain before it writes SQL — so a
deep 6+-table join is a graph traversal, not blind text-to-SQL.

## Web API (Plan 5)

The web API powers the Next.js UI. It runs on port 8000:

```bash
source backend/.venv/bin/activate
make serve-web      # uvicorn semantic_layer.web.app:app --port 8000
```

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | liveness |
| `/sources` | GET | source catalog (sql/api) |
| `/graph` | GET | schema-level knowledge graph (source/table/document nodes + REFERENCES) |
| `/chat` | POST | **SSE** stream of the agent's tool trace + a `highlight` node-set + final answer |

`/chat` emits step-level events (`tool_call`, `tool_result`, `answer`) by driving the
Plan 4 agent with `agent.stream(stream_mode="updates", subgraphs=True)`. The frontend
(`frontend/`, Next.js on port 3000) consumes this to animate the graph traversal.
