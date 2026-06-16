# NeoCarta-Local — a semantic layer over databases, APIs & documents

A locally-runnable recreation of the [Neo4j **NeoCarta**](https://github.com/neo4j-labs/neocarta)
semantic-layer pattern: a unified knowledge graph over **heterogeneous, NVIDIA-themed
data sources** that lets an AI agent answer natural-language questions **uniformly** —
without the user knowing which physical source holds the answer.

It spans all three source types:

- **Structured databases** — PostgreSQL (an 11-table normalized `sales` schema that
  supports 6+-table deep joins) and two SQLite databases (financials, org/headcount).
- **Enterprise APIs** — four mock REST services (CRM, Support/ITSM, partner inventory,
  DGX Cloud telemetry).
- **Documents** — NVIDIA press-release PDFs, parsed with **liteparse v2**.

A **deepagents** orchestrator (on OpenAI `gpt-5.4-mini`) routes each question via
graph-backed tools and delegates to three subagents — **SQL** (grounded text-to-SQL),
**API** (REST calls), and **doc** (vector RAG) — then synthesizes an answer with
provenance. A **Next.js** UI streams the agent's reasoning and **animates the path it
took through the knowledge graph**.

---

## Architecture

```
                         ┌──────────────── Neo4j knowledge graph ────────────────┐
 Postgres (Docker) ─┐    │  Metadata layer (NeoCarta): Database→Schema→Table→     │
 SQLite × 2        ─┼──► │   Column  +  REFERENCES (FK)  +  BusinessTerm          │
 Mock APIs (OpenAPI)┤    │  Document layer: Document→Chunk→Entity (POLE+O)        │
 NVIDIA PDFs ───────┘    │  Bridge + vector / full-text indexes (hybrid search)  │
   (liteparse v2)        └───────────────────────────────────────────────────────┘
                                                  ▲                    │
                            graph tools           │                    │ grounds
        ┌───────────────────────────────────────────────────────────┐ ▼
 user → │ deepagents orchestrator (gpt-5.4-mini)                     │
        │   search_catalog · get_table_schema · get_join_path · …    │
        │   ├─ sql subagent   → run_sql   (Postgres / SQLite)        │
        │   ├─ api subagent   → call_api  (CRM/ITSM/partner/DGX)     │
        │   └─ doc subagent   → search_documents (chunk vectors)     │
        └───────────────────────────────────────────────────────────┘
                                  │ SSE (tool trace + traversal highlight + answer)
                                  ▼
        Next.js split-canvas UI: chat + reasoning trace + animated graph
```

The semantic layer stores **metadata, not rows** — so the agent uses the graph to find
*where* and *how* (which tables, which join path), then fetches real values live via SQL
or REST. `get_join_path` turns a deep 6+-table join into a graph traversal over the FK
edges, which the SQL subagent then executes.

---

## Quickstart

**Prerequisites:** Docker Desktop · Python 3.11+ · Node.js 20+ · an `OPENAI_API_KEY`.

```bash
# 1. add your OpenAI key (used by the agent + document embeddings)
echo "OPENAI_API_KEY=sk-..." >> backend/.env      # backend/.env may be a symlink to ./.env

# 2. set up the platform: Docker data stores, deps, seed, ingest, backend services
./setup.sh

# 3. start the web UI
./start-ui.sh        # http://localhost:3000
```

`setup.sh` is idempotent — it starts Neo4j + Postgres, installs the Python venv, seeds
the databases, ingests the knowledge graph, and launches the **mock enterprise APIs**
(`:8001`) and the **agent web API** (`:8000`). `start-ui.sh` runs the Next.js UI (`:3000`).

**Try these in the UI** (it animates the agent's graph traversal for each):

- *Which business segment has the highest total revenue?* — deep multi-table SQL
- *How many open support tickets are there?* — the ITSM mock API
- *According to the press releases, what drove Data Center growth?* — document RAG

Stop everything: `kill $(cat logs/*.pid) 2>/dev/null; docker compose down`.

---

## What runs where

| Service | URL / port | Started by |
|---|---|---|
| Neo4j (graph) | `localhost:7687`, browser `:7474` | Docker (`setup.sh`) |
| Postgres (sales) | `localhost:5432` | Docker (`setup.sh`) |
| Mock enterprise APIs | `http://localhost:8001/docs` | `setup.sh` |
| Agent web API (SSE) | `http://localhost:8000` | `setup.sh` |
| Web UI | `http://localhost:3000` | `start-ui.sh` |

---

## Project layout

```
backend/            Python package (semantic_layer/) + tests (90 passing)
  semantic_layer/
    sources/ apis/  the data sources + 4 mock enterprise APIs
    ingest/         liteparse parsing, NeoCarta extractors, entities, glossary, embeddings, pipeline
    agent/          graph tools + sql/api/doc subagents + deepagents orchestrator
    web/            FastAPI: /sources, /graph, /chat (SSE)
  data/             deterministic NVIDIA-themed data generators + seeders
frontend/           Next.js 16 split-canvas UI (react-force-graph)
docs/               the NVIDIA PDFs + superpowers/{specs,plans} design docs
setup.sh            one-shot platform setup + backend services
start-ui.sh         start the web UI
Makefile            up · seed · ingest · serve-apis · serve-web · ask · test
docker-compose.yml  Neo4j + Postgres
```

## How it was built

Designed and implemented in five plans (full design + step-by-step plans under
`docs/superpowers/`):

1. **Data Foundation** — Dockerized Postgres + SQLite with seeded, deep-join data.
2. **Mock Enterprise APIs** — four FastAPI services with OpenAPI specs.
3. **Graph Ingestion** — NeoCarta metadata graph + liteparse documents + POLE+O
   entities + LLM glossary + embeddings, as one idempotent pipeline.
4. **deepagents Agent** — graph-backed tools + sql/api/doc subagents.
5. **Web App** — FastAPI SSE + the animated-graph Next.js UI.

## Make targets

`make up` · `make seed` · `make ingest` · `make serve-apis` · `make serve-web` ·
`make ask q="…"` (CLI agent) · `make test` (run the suite).
