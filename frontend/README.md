# NeoCarta-Local — Frontend (Plan 5: Web App)

An innovative split-canvas UI for the semantic layer: a streaming chat on the left
and a live knowledge-graph visualization on the right that **animates the path the
agent took** to answer each question.

## Prerequisites
- Node.js 20+
- The backend web API running on port 8000 (`make serve-web` from the repo root),
  which needs Neo4j + Postgres up, the graph ingested (`make ingest`), and
  `OPENAI_API_KEY` set in `backend/.env`.

## Setup
```bash
cd frontend
cp .env.local.example .env.local      # NEXT_PUBLIC_API_BASE=http://localhost:8000
npm install
npm run dev                           # http://localhost:3000
```

## What you see
- **Left:** chat with example questions; ask anything across databases, APIs, and
  documents. Below it, a live **reasoning trace** streams the agent's tool calls
  (`search_catalog`, `get_join_path`, `run_sql`, `call_api`, …) and results.
- **Right:** the knowledge graph. When you ask a question, the nodes the agent
  actually used light up (NVIDIA green / white border) while the rest dim — the
  traversal the agent took, derived from `get_join_path` / `search_catalog` /
  `search_documents` results streamed from the backend.

The chat stream is **step-level** (tool calls/results + final answer), delivered
over Server-Sent Events from `POST /chat`.

## Stack
Next.js 16 (App Router) · React · TypeScript · Tailwind · `react-force-graph-2d`.
