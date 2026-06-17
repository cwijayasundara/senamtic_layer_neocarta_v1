# Consumption Layer — Grounded Answers (SQL + Results + Summary) — Design

**Date:** 2026-06-17
**Status:** Draft (for review)
**Builds on:** [Fiscal-Period Auto-Scoping](2026-06-17-fiscal-period-auto-scoping-design.md),
[Value Layer + Document Context Graph](2026-06-16-value-layer-and-doc-context-graph-design.md),
[NeoCarta-Local Semantic Layer](2026-06-16-neocarta-local-semantic-layer-design.md)

## Goal

Make the **Consumption Layer** a faithful *functional* clone of the
["Semantic Layer Architecture" diagram](https://neo4j.com/blog/genai/build-a-semantic-layer-from-gcp-with-neocarta/):
every answer surfaces **(1) the generated SQL/API calls, (2) a results table, and
(3) the natural-language summary** — with a groundedness guard that flags numbers in the
summary that don't appear in any retrieved result.

Scope decisions already locked (2026-06-17):
- **Agent layer:** keep the existing deepagents direct-tool orchestration. **No MCP servers.**
- **Data source layer:** keep Postgres + SQLite + mock REST APIs. **No GCP/BigQuery.**

So the remaining gap to match the diagram is the Consumption Layer alone.

## Problem

Today the final answer is a single plain-text string. `stream_chat_events`
(`web/events.py:54`) emits `{type:"answer", content, highlight}` and `ChatPanel.tsx:49-51`
renders `content` as `whitespace-pre-wrap` text. The generated SQL, the rows it returned,
and the document citations exist only inside the raw `tool_result` stream — truncated to
4000 chars and dumped as mono-text in `TracePanel.tsx:15-20`. Three consequences:

1. **Diagram mismatch.** The blog's signature view (SQL block → results table → summary)
   is absent; a user can't see the query or the table behind a number.
2. **Groundedness is prompt-only.** The SQL subagent is *asked* to "Report the rows and the
   SQL you ran" (`build.py:84`), but nothing captures or verifies it. The orchestrator's
   final synthesis can state a number that never appeared in a result row, and nothing
   catches it.
3. **No structured provenance.** Citations like `doc:NVIDIAAn_2026` are plain text, not
   linked to the chunk that was actually retrieved.

The fix is one structured payload assembled in `events.py` and a consumption-layer UI that
renders it — plus a lightweight numeric grounding check over the captured rows.

## Approach: capture structured provenance at the stream, verify, render

`events.py` already sees every `tool_call` (with args) and every `ToolMessage` (with the
result). It currently throws the useful parts away. Instead, **pair each tool call with its
result by `tool_call_id`** and accumulate three structured collections, then run a numeric
grounding check against the captured rows before emitting a richer `answer` event.

No agent/graph changes are required — this is pure consumption-layer plumbing over the
existing stream. Prompts stay as-is (the subagents already produce SQL and citations).

### New `answer` event shape

```jsonc
{
  "type": "answer",
  "content": "…natural-language summary…",      // unchanged
  "highlight": ["table:…", "doc:…"],            // unchanged
  "sql_runs": [                                  // new
    { "source": "sales_pg", "sql": "SELECT …", "columns": ["customer","q","revenue"],
      "rows": [["Acme","FY2027-Q1",1234567], …], "row_count": 12, "error": null }
  ],
  "api_calls": [                                 // new
    { "source": "itsm", "path": "/tickets", "params": {"status":"open"}, "row_count": 8 }
  ],
  "doc_citations": [                             // new
    { "doc_id": "doc:NVIDIAAn_2026", "chunk_id": "doc:NVIDIAAn_2026:chunk:3",
      "quote": "Q1 Fiscal 2027 Summary… Revenue $81,615M", "score": 0.82 }
  ],
  "caveats": [                                   // new — from the grounding check
    "The summary cites $81,615M, which was not found in any SQL/API result row." ]
}
```

Backwards compatible: `content` + `highlight` are unchanged, so the graph canvas and any
existing consumers keep working; new fields are additive.

### Components

1. **`web/events.py` — pair calls with results, accumulate provenance.**
   The orchestrator and subagents stream as subgraphs (`subgraphs=True`), so tool calls and
   their `ToolMessage` results arrive as separate messages. Track a
   `pending: dict[call_id, {name, args}]` from `AIMessage.tool_calls` (each has an `id`),
   and on each `ToolMessage` look up `m.tool_call_id` to recover the call's args. Then:
   - `run_sql` → append to `sql_runs` using `args["source"]`, `args["sql"]`, and the parsed
     `{columns, rows}` from the result (or `error` if the result is an error payload).
     Truncate `rows` to `settings.agent_max_rows` (already enforced server-side) and cap the
     serialized cell length so the event stays small.
   - `call_api` → append to `api_calls` using `args["source"]`, `args["path"]`,
     `args["params"]`, and `row_count` from the JSON length.
   - `search_documents` → append to `doc_citations` (dedupe by `chunk_id`), keeping the
     single highest-scoring quote per `doc_id` for display.
   Existing `_collect_highlight` is unchanged and runs alongside.

2. **`web/grounding.py` — `check_numeric_grounding(content, sql_runs, api_calls)`** (new).
   A guard, not a proof. Extract numeric tokens from `content` via regex
   (`\$?\d[\d,]*(?:\.\d+)?%?`), normalize each (strip `$ , %`, fold to float). Build the set
   of numeric values appearing in any captured cell (same normalization; also derive
   percentages and rounded forms to tolerate "38.81%" vs `0.3881`/`38.81`). For every prose
   number with magnitude ≥ a threshold (skip years 2024–2027 and small ordinals like "3
   tables" to cut noise) that has **no** match in the cell set, emit one caveat. Returns
   `list[str]`. Honest limits documented inline: it catches fabricated/transcription-wrong
   figures, not wrong-but-present ones, and may false-positive on derived numbers (ratios the
   LLM computed correctly from two cells) — caveats are advisory, never block the answer.

3. **`lib/types.ts` — extend `ChatEvent`'s `answer` variant** with `sql_runs`, `api_calls`,
   `doc_citations`, `caveats` (mirror the JSON above as TS types: `SqlRun`, `ApiCall`,
   `DocCitation`).

4. **`components/AnswerPanel.tsx`** (new) — renders one answered question as the blog does:
   - **Summary** (existing `content`, `whitespace-pre-wrap`).
   - **Caveats** banner (amber) when `caveats.length > 0`.
   - **Generated SQL** — one collapsible `<pre>` per `sql_runs[i]` with a source chip; show
     `error` in red when present.
   - **Results table** — render `columns`/`rows` as a real `<table>` per `sql_run` (and a
     compact summary for `api_calls`).
   - **Citations** — `doc_citations` as chips; clicking a chip calls the existing node
     inspector with `chunk_id`/`doc_id` (reuse `NodeDetails` selection), tying the
     consumption layer back to the graph.

5. **`components/ChatPanel.tsx`** — replace the lone `answer` text block (`:49-51`) with
   `<AnswerPanel .../>`, fed by the enriched event. `useChatStream.ts` already collects
   events; expose the full final `answer` event (not just its `content` string) to the panel.

### Data flow (additions)

```
agent.stream (unchanged)
  → events.py: per ToolMessage, pair by tool_call_id with pending call args
      run_sql        → sql_runs[]
      call_api       → api_calls[]
      search_documents → doc_citations[]
  → final AIMessage text → content
  → grounding.check_numeric_grounding(content, sql_runs, api_calls) → caveats[]
  → yield answer{content, highlight, sql_runs, api_calls, doc_citations, caveats}
  → AnswerPanel renders summary + caveats + SQL + results table + citations
```

## Testing

- **events pairing** (`test_web_chat_events.py`, extend): a chat run that hits SQL yields an
  `answer` event whose `sql_runs[0]` has non-empty `sql`, `columns`, and `rows`, and whose
  `sql` matches the `run_sql` tool-call args (proves call↔result pairing by `tool_call_id`).
- **doc citations**: a document question yields `doc_citations` with a `chunk_id` and a
  non-empty `quote`; deduped to one entry per `doc_id`.
- **grounding — true positive**: `check_numeric_grounding("Revenue was $999,999M.", [], [])`
  → one caveat (number absent from any row).
- **grounding — true negative**: a `content` whose numbers all appear in `sql_runs[0].rows`
  → `[]` (no false caveat); include the `38.81%` ↔ cell-`0.3881` normalization case.
- **grounding — noise filter**: "FY2027" and "3 tables" do not produce caveats.
- **frontend** (component test): given an `answer` event with a `sql_runs` entry, `AnswerPanel`
  renders a `<table>` with the right column headers and a `<pre>` containing the SQL; given
  `caveats`, renders the amber banner.

Run after backend changes: `make ingest` (per the standing note — pytest wipes chunk
embeddings, so doc-citation tests need re-ingest first).

## Out of scope (YAGNI)

- MCP servers (Neocarta MCP / SQL MCP) — explicitly deferred; revisit only if a faithful
  *structural* clone is later wanted.
- BigQuery/Dataplex sources and query-log → JOIN inference.
- `Glossary → Category → BusinessTerm` hierarchy (current flat `BusinessTerm` is kept).
- LLM-based or entailment groundedness scoring; confidence percentages. The numeric check is
  a deterministic guard, not a model.
- Editable/re-runnable SQL from the UI (read-only display only).
```
