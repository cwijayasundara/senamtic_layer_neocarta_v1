# Plan 5: Web App — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Backend tasks are full TDD (pytest); frontend tasks gate on `tsc`/`next build` + a Playwright smoke (no unit-test framework is added).

**Goal:** A distinctive web UI that makes the semantic layer *visible*: a split canvas with a streaming chat on the left and a live knowledge-graph visualization on the right that **animates the path the agent took** to answer each question — lighting up business terms → tables/endpoints → the `get_join_path` chain → source (or `Document → Chunk` for document answers) — alongside a reasoning-trace panel that streams the agent's tool calls, SQL/REST, and citations.

**Architecture:** A FastAPI web API (`semantic_layer/web/`) exposes `GET /sources`, `GET /graph` (schema-level graph for the canvas), and `POST /chat` (Server-Sent Events). The `/chat` handler drives the Plan 4 agent via `agent.stream(input, stream_mode="updates", subgraphs=True)`, translating each tool call/result into an SSE event and accumulating a `highlight` node-set (from `get_join_path`/`search_catalog`/`search_documents` results) emitted with the final answer. A **Next.js** (App Router, TypeScript, Tailwind) frontend renders the graph with `react-force-graph-2d`, consumes the SSE stream, shows the reasoning trace, and animates the highlighted traversal.

**Tech Stack:** Backend — FastAPI, `sse-starlette` (or manual SSE), the Plan 4 agent. Frontend — Next.js 15, React 19, TypeScript, Tailwind CSS, `react-force-graph-2d`, native `fetch` streaming. Node 20+. Builds on Plans 1–4.

**Prerequisites:** Plans 1–4 merged. Neo4j + Postgres up, databases seeded, graph ingested + embedded (`make ingest`), `OPENAI_API_KEY` set. Node.js 20+ for the frontend.

This is sub-plan 5 of 5 (Data Foundation → Mock APIs → Graph Ingestion → Agent → **Web App**) — the final plan.

---

## Verified streaming behavior (confirmed against the live Plan 4 agent — do not re-derive)

`agent.stream({"messages":[{"role":"user","content": q}]}, stream_mode="updates", subgraphs=True)` yields `(namespace_tuple, {node_name: {"messages": [...]}})`. Across the stream:
- `AIMessage.tool_calls` → list of `{name, args, id}` (the tool the agent decided to call).
- `ToolMessage` → `.name` + `.content` (the tool's JSON result string).
- The orchestrator's delegation appears as a `task` tool call; the subagent's internal tools (`get_table_schema`, `run_sql`, `get_join_path`, `call_api`, `search_documents`) appear under a non-root namespace (`ns[-1]` like `tools:<uuid>`) because of `subgraphs=True`.
- The final assistant answer is the last `AIMessage` with non-empty `content` and no tool calls.

Confirmed tool sequence for a structured question: `list_sources → search_catalog → get_table_schema×N → get_join_path → task → (subagent) get_table_schema×N → run_sql → final answer`.

---

## File Structure

```
backend/
  pyproject.toml                          # (modify) add sse-starlette
  semantic_layer/web/
    __init__.py
    graph_api.py                          # schema-level graph + sources queries
    events.py                             # agent.stream -> SSE event dicts + highlight extraction
    app.py                                # FastAPI app: /health, /sources, /graph, /chat (SSE), CORS
  tests/
    test_web_graph_api.py                 # neo4j
    test_web_chat_events.py               # neo4j + postgres + openai
Makefile                                  # (modify) add serve-web
frontend/                                 # Next.js app (new)
  package.json, tsconfig.json, next.config.ts, tailwind.config.ts, postcss.config.mjs
  .env.local.example                      # NEXT_PUBLIC_API_BASE
  app/
    layout.tsx, globals.css, page.tsx     # split-canvas shell
  components/
    GraphCanvas.tsx                       # react-force-graph-2d + highlight animation
    ChatPanel.tsx                         # messages + input
    TracePanel.tsx                        # streamed tool trace
  lib/
    api.ts                                # fetch /sources, /graph
    useChatStream.ts                      # SSE consumption hook
    types.ts
README.md (frontend)                      # run instructions
backend/README.md                         # (modify) document the web API
```

---

## Task 1: Web API — `/sources` and `/graph` (schema-level graph)

**Files:**
- Modify: `backend/pyproject.toml` (add `"sse-starlette>=2.0"`)
- Create: `backend/semantic_layer/web/__init__.py` (empty)
- Create: `backend/semantic_layer/web/graph_api.py`
- Create: `backend/semantic_layer/web/app.py`
- Test: `backend/tests/test_web_graph_api.py`

- [ ] **Step 1: Add dep** to `pyproject.toml` dependencies: `"sse-starlette>=2.0"`. Install: `cd backend && ./.venv/bin/python -m pip install -e ".[dev]"`.

- [ ] **Step 2: Write the failing test** `backend/tests/test_web_graph_api.py`

```python
import pytest
from fastapi.testclient import TestClient

from semantic_layer.web.app import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.neo4j
def test_sources_endpoint(ingested_graph):
    data = client.get("/sources").json()
    names = {s["name"] for s in data}
    assert {"sales_pg", "crm"} <= names


@pytest.mark.neo4j
def test_graph_endpoint_returns_tables_and_refs(ingested_graph):
    g = client.get("/graph").json()
    assert "nodes" in g and "edges" in g
    kinds = {n["kind"] for n in g["nodes"]}
    assert {"source", "table"} <= kinds
    # the deep sales schema contributes table-level REFERENCES edges
    assert any(e["type"] == "REFERENCES" for e in g["edges"])
    # document node present
    assert any(n["kind"] == "document" for n in g["nodes"])
```

- [ ] **Step 3: Implement `backend/semantic_layer/web/graph_api.py`**

```python
"""Read-only graph projections for the web UI (schema-level, renderable)."""

from semantic_layer.agent.driver import driver
from semantic_layer.config import settings

_SQL_PLATFORMS = {"POSTGRESQL", "SQLITE"}


def get_sources() -> list[dict]:
    rows = driver().execute_query(
        "MATCH (d:Database) RETURN d.name AS name, d.platform AS platform ORDER BY name",
        database_=settings.neo4j_database,
    ).records
    out = []
    for r in rows:
        platform = (r["platform"] or "").upper()
        out.append({"name": r["name"], "platform": platform,
                    "kind": "sql" if platform in _SQL_PLATFORMS else "api"})
    return out


def get_schema_graph() -> dict:
    """Source + table + document nodes; HAS_TABLE, REFERENCES (table-level), MENTIONS-free."""
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    db_rows = driver().execute_query(
        "MATCH (d:Database) RETURN d.id AS id, d.name AS name, d.platform AS platform",
        database_=settings.neo4j_database,
    ).records
    for r in db_rows:
        platform = (r["platform"] or "").upper()
        nodes[r["id"]] = {"id": r["id"], "label": r["name"], "kind": "source",
                          "source": r["name"],
                          "platform": "sql" if platform in _SQL_PLATFORMS else "api"}

    tbl_rows = driver().execute_query(
        """
        MATCH (d:Database)-[:HAS_SCHEMA]->(:Schema)-[:HAS_TABLE]->(t:Table)
        RETURN t.id AS id, t.name AS name, d.id AS db_id, d.name AS source
        """,
        database_=settings.neo4j_database,
    ).records
    for r in tbl_rows:
        nodes[r["id"]] = {"id": r["id"], "label": r["name"], "kind": "table",
                          "source": r["source"]}
        edges.append({"source": r["db_id"], "target": r["id"], "type": "HAS_TABLE"})

    ref_rows = driver().execute_query(
        """
        MATCH (t1:Table)-[:HAS_COLUMN]->(:Column)-[:REFERENCES]->(:Column)<-[:HAS_COLUMN]-(t2:Table)
        WHERE t1 <> t2
        RETURN DISTINCT t1.id AS a, t2.id AS b
        """,
        database_=settings.neo4j_database,
    ).records
    for r in ref_rows:
        edges.append({"source": r["a"], "target": r["b"], "type": "REFERENCES"})

    doc_rows = driver().execute_query(
        "MATCH (d:Document) RETURN d.id AS id, d.title AS title",
        database_=settings.neo4j_database,
    ).records
    for r in doc_rows:
        nodes[r["id"]] = {"id": r["id"], "label": r["title"], "kind": "document",
                          "source": "documents"}

    return {"nodes": list(nodes.values()), "edges": edges}
```

- [ ] **Step 4: Implement `backend/semantic_layer/web/app.py`** (the `/chat` SSE route is added in Task 2; for now just health/sources/graph + CORS)

```python
"""FastAPI web API for the semantic-layer UI."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from semantic_layer.web.graph_api import get_sources, get_schema_graph

app = FastAPI(title="NeoCarta-Local Web API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/sources")
def sources():
    return get_sources()


@app.get("/graph")
def graph():
    return get_schema_graph()
```

- [ ] **Step 5: Run** `cd backend && ./.venv/bin/python -m pytest tests/test_web_graph_api.py -v` → 3 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/pyproject.toml backend/semantic_layer/web/__init__.py backend/semantic_layer/web/graph_api.py backend/semantic_layer/web/app.py backend/tests/test_web_graph_api.py
git commit -m "feat(web): FastAPI /sources + /graph schema-graph endpoints with CORS"
```

---

## Task 2: SSE `/chat` — stream agent trace + highlight set

**Files:**
- Create: `backend/semantic_layer/web/events.py`
- Modify: `backend/semantic_layer/web/app.py`
- Test: `backend/tests/test_web_chat_events.py`

- [ ] **Step 1: Write the failing test** `backend/tests/test_web_chat_events.py`

```python
import json

import pytest

from semantic_layer.web.events import stream_chat_events


@pytest.mark.neo4j
@pytest.mark.postgres
@pytest.mark.openai
def test_stream_emits_tools_and_final_answer(ingested_graph, require_openai):
    events = list(stream_chat_events("Which segment has the most revenue? Use the sales database."))
    types = [e["type"] for e in events]
    assert "tool_call" in types
    assert types[-1] == "answer"
    answer = events[-1]
    assert "Data Center" in answer["content"]
    # the join path / catalog hits produced a highlight set of graph node ids
    assert isinstance(answer["highlight"], list)
    assert any(nid.startswith("table:sales_pg") for nid in answer["highlight"])
    # a run_sql result was surfaced in the trace
    assert any(e["type"] == "tool_result" and e["name"] == "run_sql" for e in events)
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `backend/semantic_layer/web/events.py`**

```python
"""Translate the agent's streamed steps into UI events + a highlight node-set."""

import json
from typing import Iterator

from semantic_layer.agent.build import build_agent


def _collect_highlight(tool_name: str, content: str, highlight: set[str]) -> None:
    """Accumulate graph node ids to light up, from specific tool results."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return
    if tool_name == "get_join_path" and isinstance(data, dict):
        for tid in data.get("tables", []):
            highlight.add(tid)
    elif tool_name == "search_catalog" and isinstance(data, list):
        for hit in data:
            tid = hit.get("table_id", "")
            if tid.startswith("table:"):
                highlight.add(tid)
    elif tool_name == "search_documents" and isinstance(data, list):
        for hit in data:
            if hit.get("doc_id"):
                highlight.add(hit["doc_id"])


def stream_chat_events(question: str) -> Iterator[dict]:
    """Yield UI events: {type: tool_call|tool_result|answer, ...}."""
    agent = build_agent()
    highlight: set[str] = set()
    final = ""
    for ns, chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="updates", subgraphs=True,
    ):
        scope = ns[-1].split(":")[0] if ns else "root"
        for _node, update in chunk.items():
            messages = update.get("messages", []) if isinstance(update, dict) else []
            for m in messages:
                for call in getattr(m, "tool_calls", None) or []:
                    yield {"type": "tool_call", "scope": scope,
                           "name": call.get("name"), "args": call.get("args", {})}
                if type(m).__name__ == "ToolMessage":
                    name = getattr(m, "name", "")
                    content = str(getattr(m, "content", ""))
                    _collect_highlight(name, content, highlight)
                    yield {"type": "tool_result", "scope": scope,
                           "name": name, "content": content[:4000]}
                elif type(m).__name__ == "AIMessage" and getattr(m, "content", None) \
                        and not getattr(m, "tool_calls", None):
                    final = m.content
    yield {"type": "answer", "content": final, "highlight": sorted(highlight)}
```

- [ ] **Step 4: Add the SSE route to `backend/semantic_layer/web/app.py`**

```python
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from semantic_layer.web.events import stream_chat_events


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
async def chat(req: ChatRequest):
    def event_generator():
        for event in stream_chat_events(req.question):
            yield {"event": event["type"], "data": json.dumps(event)}
    return EventSourceResponse(event_generator())
```

Add `import json` at the top of `app.py`.

- [ ] **Step 5: Run** `cd backend && ./.venv/bin/python -m pytest tests/test_web_chat_events.py -v` (live agent run; slow). Expected 1 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/semantic_layer/web/events.py backend/semantic_layer/web/app.py backend/tests/test_web_chat_events.py
git commit -m "feat(web): SSE /chat streams agent trace + traversal highlight set"
```

---

## Task 3: Run target + backend docs

**Files:** Modify `Makefile`, `backend/README.md`.

- [ ] **Step 1:** Add `serve-web` to `.PHONY` and a target (TAB-indented):

```makefile
serve-web:
	cd backend && uvicorn semantic_layer.web.app:app --port 8000 --reload
```

Verify `make -n serve-web`.

- [ ] **Step 2:** Append a "Web API" section to `backend/README.md` documenting `make serve-web` (port 8000), the three endpoints, and that `/chat` is SSE. Commit.

```bash
git add Makefile backend/README.md
git commit -m "feat(web): make serve-web target + web API docs"
```

---

## Task 4: Frontend scaffold (Next.js + Tailwind) + API client

**Files:** `frontend/` Next.js app.

- [ ] **Step 1: Scaffold** from the repo root:
`npx create-next-app@latest frontend --typescript --tailwind --app --no-src-dir --import-alias "@/*" --eslint --use-npm`
Then `cd frontend && npm install react-force-graph-2d`.

- [ ] **Step 2: Create `frontend/.env.local.example`**

```bash
NEXT_PUBLIC_API_BASE=http://localhost:8000
```

- [ ] **Step 3: Create `frontend/lib/types.ts`**

```typescript
export type GraphNode = { id: string; label: string; kind: "source" | "table" | "document"; source: string; platform?: string };
export type GraphEdge = { source: string; target: string; type: string };
export type GraphData = { nodes: GraphNode[]; edges: GraphEdge[] };
export type Source = { name: string; platform: string; kind: "sql" | "api" };

export type ChatEvent =
  | { type: "tool_call"; scope: string; name: string; args: Record<string, unknown> }
  | { type: "tool_result"; scope: string; name: string; content: string }
  | { type: "answer"; content: string; highlight: string[] };
```

- [ ] **Step 4: Create `frontend/lib/api.ts`**

```typescript
import type { GraphData, Source } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export async function fetchSources(): Promise<Source[]> {
  const r = await fetch(`${BASE}/sources`);
  return r.json();
}

export async function fetchGraph(): Promise<GraphData> {
  const r = await fetch(`${BASE}/graph`);
  return r.json();
}

export const API_BASE = BASE;
```

- [ ] **Step 5: Type-check** `cd frontend && npx tsc --noEmit` → no errors. **Build** `npm run build` → succeeds.

- [ ] **Step 6: Commit**

```bash
git add frontend
git commit -m "feat(web): Next.js frontend scaffold + API client and types"
```

(Ensure the root `.gitignore` already ignores `node_modules/` and `.next/` — it does from Plan 1.)

---

## Task 5: `useChatStream` SSE hook

**Files:** `frontend/lib/useChatStream.ts`

- [ ] **Step 1: Implement `frontend/lib/useChatStream.ts`**

```typescript
"use client";
import { useCallback, useState } from "react";
import { API_BASE } from "./api";
import type { ChatEvent } from "./types";

export function useChatStream() {
  const [events, setEvents] = useState<ChatEvent[]>([]);
  const [answer, setAnswer] = useState<string>("");
  const [highlight, setHighlight] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);

  const ask = useCallback(async (question: string) => {
    setEvents([]); setAnswer(""); setHighlight([]); setBusy(true);
    const resp = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const reader = resp.body!.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const frames = buf.split("\n\n");
      buf = frames.pop() ?? "";
      for (const frame of frames) {
        const line = frame.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const evt = JSON.parse(line.slice(5).trim()) as ChatEvent;
        setEvents((prev) => [...prev, evt]);
        if (evt.type === "answer") { setAnswer(evt.content); setHighlight(evt.highlight); }
      }
    }
    setBusy(false);
  }, []);

  return { events, answer, highlight, busy, ask };
}
```

- [ ] **Step 2: Type-check** `npx tsc --noEmit` → no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/lib/useChatStream.ts
git commit -m "feat(web): useChatStream SSE consumption hook"
```

---

## Task 6: GraphCanvas component (react-force-graph + highlight)

**Files:** `frontend/components/GraphCanvas.tsx`

- [ ] **Step 1: Implement `frontend/components/GraphCanvas.tsx`**

```tsx
"use client";
import dynamic from "next/dynamic";
import { useMemo } from "react";
import type { GraphData } from "@/lib/types";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

const KIND_COLOR: Record<string, string> = {
  source: "#76b900", // NVIDIA green
  table: "#3b82f6",
  document: "#f59e0b",
};

export function GraphCanvas({ graph, highlight }: { graph: GraphData; highlight: string[] }) {
  const hi = useMemo(() => new Set(highlight), [highlight]);
  const data = useMemo(() => ({
    nodes: graph.nodes.map((n) => ({ ...n })),
    links: graph.edges.map((e) => ({ source: e.source, target: e.target, type: e.type })),
  }), [graph]);

  return (
    <ForceGraph2D
      graphData={data}
      backgroundColor="#0b0f14"
      nodeRelSize={5}
      linkColor={(l: any) => (hi.has(typeof l.source === "object" ? l.source.id : l.source) &&
        hi.has(typeof l.target === "object" ? l.target.id : l.target) ? "#76b900" : "#1f2937")}
      linkWidth={(l: any) => (hi.has(typeof l.source === "object" ? l.source.id : l.source) &&
        hi.has(typeof l.target === "object" ? l.target.id : l.target) ? 3 : 1)}
      nodeCanvasObject={(node: any, ctx, scale) => {
        const active = hi.has(node.id);
        const r = active ? 7 : 4;
        ctx.beginPath();
        ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
        ctx.fillStyle = KIND_COLOR[node.kind] ?? "#9ca3af";
        ctx.globalAlpha = active || hi.size === 0 ? 1 : 0.25;
        ctx.fill();
        if (active) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke(); }
        if (active || scale > 2) {
          ctx.globalAlpha = 1; ctx.fillStyle = "#e5e7eb";
          ctx.font = `${10 / scale}px sans-serif`;
          ctx.fillText(node.label, node.x + r + 1, node.y + 3);
        }
        ctx.globalAlpha = 1;
      }}
    />
  );
}
```

- [ ] **Step 2: Type-check** `npx tsc --noEmit` (react-force-graph types are loose; `any` on callbacks is expected). Build `npm run build`.

- [ ] **Step 3: Commit**

```bash
git add frontend/components/GraphCanvas.tsx
git commit -m "feat(web): GraphCanvas with kind colors and traversal highlight"
```

---

## Task 7: ChatPanel + TracePanel

**Files:** `frontend/components/ChatPanel.tsx`, `frontend/components/TracePanel.tsx`

- [ ] **Step 1: Implement `frontend/components/ChatPanel.tsx`**

```tsx
"use client";
import { useState } from "react";

export function ChatPanel({ answer, busy, onAsk }:
  { answer: string; busy: boolean; onAsk: (q: string) => void }) {
  const [q, setQ] = useState("");
  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-auto p-4 space-y-3">
        {answer && (
          <div className="rounded-lg bg-gray-800 p-3 text-gray-100 whitespace-pre-wrap">{answer}</div>
        )}
        {busy && <div className="text-sm text-gray-400 animate-pulse">thinking…</div>}
      </div>
      <form
        className="p-3 border-t border-gray-800 flex gap-2"
        onSubmit={(e) => { e.preventDefault(); if (q.trim()) { onAsk(q); setQ(""); } }}
      >
        <input
          className="flex-1 rounded bg-gray-900 border border-gray-700 px-3 py-2 text-gray-100"
          placeholder="Ask across databases, APIs, and documents…"
          value={q} onChange={(e) => setQ(e.target.value)} disabled={busy}
        />
        <button className="rounded bg-[#76b900] px-4 py-2 font-medium text-black disabled:opacity-50"
          disabled={busy}>Ask</button>
      </form>
    </div>
  );
}
```

- [ ] **Step 2: Implement `frontend/components/TracePanel.tsx`**

```tsx
"use client";
import type { ChatEvent } from "@/lib/types";

export function TracePanel({ events }: { events: ChatEvent[] }) {
  return (
    <div className="h-full overflow-auto p-3 text-xs font-mono space-y-1">
      {events.map((e, i) => {
        if (e.type === "tool_call")
          return <div key={i} className="text-blue-300">→ {e.name}({JSON.stringify(e.args)})</div>;
        if (e.type === "tool_result")
          return <div key={i} className="text-gray-400 truncate">{e.name}: {e.content}</div>;
        return <div key={i} className="text-[#76b900]">✓ answer ({e.highlight.length} nodes lit)</div>;
      })}
    </div>
  );
}
```

- [ ] **Step 3: Type-check + build.** Commit.

```bash
git add frontend/components/ChatPanel.tsx frontend/components/TracePanel.tsx
git commit -m "feat(web): ChatPanel and streamed TracePanel components"
```

---

## Task 8: Split-canvas page wiring

**Files:** `frontend/app/page.tsx`, `frontend/app/globals.css` (minor)

- [ ] **Step 1: Implement `frontend/app/page.tsx`**

```tsx
"use client";
import { useEffect, useState } from "react";
import { fetchGraph } from "@/lib/api";
import { useChatStream } from "@/lib/useChatStream";
import { GraphCanvas } from "@/components/GraphCanvas";
import { ChatPanel } from "@/components/ChatPanel";
import { TracePanel } from "@/components/TracePanel";
import type { GraphData } from "@/lib/types";

export default function Home() {
  const [graph, setGraph] = useState<GraphData>({ nodes: [], edges: [] });
  const { events, answer, highlight, busy, ask } = useChatStream();

  useEffect(() => { fetchGraph().then(setGraph).catch(() => {}); }, []);

  return (
    <main className="h-screen w-screen grid grid-cols-[420px_1fr] bg-[#0b0f14] text-gray-100">
      <section className="border-r border-gray-800 flex flex-col min-h-0">
        <header className="px-4 py-3 border-b border-gray-800 font-semibold">
          NeoCarta-Local <span className="text-[#76b900]">semantic layer</span>
        </header>
        <div className="flex-1 min-h-0"><ChatPanel answer={answer} busy={busy} onAsk={ask} /></div>
        <div className="h-48 border-t border-gray-800"><TracePanel events={events} /></div>
      </section>
      <section className="min-h-0"><GraphCanvas graph={graph} highlight={highlight} /></section>
    </main>
  );
}
```

- [ ] **Step 2: Build** `npm run build` → succeeds. Commit.

```bash
git add frontend/app/page.tsx frontend/app/globals.css
git commit -m "feat(web): split-canvas page wiring chat + trace + animated graph"
```

---

## Task 9: End-to-end smoke (Playwright), README, run instructions

**Files:** `frontend/README.md`, root run docs.

- [ ] **Step 1: Manual/automated smoke.** With Neo4j + Postgres up, the graph ingested, and `OPENAI_API_KEY` set: start the web API (`make serve-web`) and the frontend (`cd frontend && npm run dev`). Using the Playwright MCP tools: navigate to `http://localhost:3000`, confirm the graph canvas renders source + table nodes, type a question ("Which segment has the most revenue?"), submit, and confirm (a) trace events stream into the TracePanel, (b) an answer appears in the ChatPanel, and (c) graph nodes light up (the highlighted traversal). Capture a screenshot.

- [ ] **Step 2: Write `frontend/README.md`** with prerequisites (backend running via `make serve-web`, `cp .env.local.example .env.local`), `npm install`, `npm run dev`, and a note that the right canvas animates the agent's traversal for each question.

- [ ] **Step 3: Append a "Web UI" section to `backend/README.md`** linking the two processes (API on 8000, Next.js on 3000) and listing the four example questions from Plan 4.

- [ ] **Step 4: Run the backend full suite** `cd backend && ./.venv/bin/python -m pytest -q` and confirm no regressions. Commit.

```bash
git add frontend/README.md backend/README.md
git commit -m "docs(web): frontend run instructions, smoke verification, and UI docs"
```

---

## Self-Review

**Spec coverage (Plan 5 scope):** Next.js (App Router) + React + Tailwind frontend (Tasks 4–8) ✓; split-canvas — chat left, graph right (Task 8) ✓; **animated graph traversal** lighting up the agent's path, derived from `get_join_path`/`search_catalog`/`search_documents` results in the `highlight` set (Tasks 2, 6) ✓; reasoning-trace panel streaming tool calls/results (Tasks 2, 7) ✓; FastAPI SSE `/chat` + `/graph` + `/sources` with CORS (Tasks 1–2) ✓; run targets + docs (Tasks 3, 9) ✓. True token-by-token streaming is out of scope — the stream is step-level (tool calls/results + final answer), which is the meaningful trace; note this in the README rather than implying token streaming.

**External-API honesty:** the SSE backend uses the verified `agent.stream(stream_mode="updates", subgraphs=True)` shape (tool_calls on AIMessage, ToolMessage name/content, subagent namespace) confirmed against the live Plan 4 agent. `react-force-graph-2d` is loaded via `next/dynamic` with `ssr:false` (it touches `window`).

**Testability split:** backend tasks (1–2) are full pytest TDD against the live graph/agent; frontend tasks (4–8) gate on `tsc --noEmit` + `next build`; the integrated behavior is verified by the Playwright smoke (Task 9). This is honest — no fake unit tests are added for the React layer.

**Type/name consistency:** `GraphNode.kind` values (`source`/`table`/`document`) produced by `graph_api.get_schema_graph` match `GraphCanvas` `KIND_COLOR` and the `test_web_graph_api` assertions. `ChatEvent` union (`tool_call`/`tool_result`/`answer`) emitted by `events.stream_chat_events` matches `useChatStream` parsing and `TracePanel` rendering. `highlight` node ids are graph node ids (`table:*`, `doc:*`) that exist in `/graph` output, so the canvas can match them. `NEXT_PUBLIC_API_BASE` used consistently in `api.ts` and `useChatStream.ts`.

**Scope check:** one coherent subsystem (the web app over the Plan 4 agent). 9 tasks; backend independently testable, frontend build-verifiable + smoke-tested. This completes the 5-plan arc.
```
