# Consumption Layer — Grounded Answers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface, for every answer, the generated SQL/API calls, a results table, and the NL summary — with a deterministic guard that flags summary numbers absent from any retrieved result.

**Architecture:** Pure consumption-layer plumbing over the existing agent stream. `web/events.py` pairs each tool call with its result by `tool_call_id` and accumulates structured `sql_runs` / `api_calls` / `doc_citations`; a new `web/grounding.py` checks the summary's numbers against those results and emits advisory `caveats`. The enriched `answer` event is rendered by a new `AnswerPanel.tsx`. No agent, graph, prompt, or data-source changes.

**Tech Stack:** Python 3.14 + pytest (backend), FastAPI SSE, Next.js 16 / React 19 / TypeScript (frontend, no test runner — verify with `npm run build` + manual run).

**Spec:** [Consumption Layer — Grounded Answers](../specs/2026-06-17-consumption-layer-grounded-answers-design.md)

---

## File Structure

- **Create** `backend/semantic_layer/web/grounding.py` — `check_numeric_grounding()`, pure/deterministic.
- **Create** `backend/tests/test_grounding.py` — unit tests for the guard.
- **Create** `backend/tests/test_provenance.py` — unit tests for the `_Provenance` collector.
- **Modify** `backend/semantic_layer/web/events.py` — add `_Provenance`, pair calls↔results, enrich `answer`.
- **Modify** `backend/tests/test_web_chat_events.py` — assert the enriched `answer` fields end-to-end.
- **Modify** `frontend/lib/types.ts` — extend the `answer` `ChatEvent` variant; add `SqlRun`/`ApiCall`/`DocCitation`/`AnswerEvent`.
- **Modify** `frontend/lib/useChatStream.ts` — expose the full final `answer` event.
- **Create** `frontend/components/AnswerPanel.tsx` — summary + caveats + SQL + results table + citations.
- **Modify** `frontend/components/ChatPanel.tsx` — render `<AnswerPanel>` instead of the plain text block.
- **Modify** `frontend/app/page.tsx` — pass `answerEvent` to `ChatPanel`.

---

## Task 1: Numeric grounding guard (`grounding.py`)

**Files:**
- Create: `backend/semantic_layer/web/grounding.py`
- Test: `backend/tests/test_grounding.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_grounding.py
from semantic_layer.web.grounding import check_numeric_grounding


def _sql(rows, columns=("v",)):
    return [{"source": "sales_pg", "sql": "SELECT …", "columns": list(columns),
             "rows": rows, "row_count": len(rows), "error": None}]


def test_flags_number_absent_from_rows():
    caveats = check_numeric_grounding("Revenue was $999,999M.", _sql([[12345]]), [], [])
    assert len(caveats) == 1
    assert "999,999" in caveats[0]


def test_no_caveat_when_number_present():
    caveats = check_numeric_grounding("Total revenue was 1234567.", _sql([[1234567]]), [], [])
    assert caveats == []


def test_percentage_fraction_normalization():
    # prose says 38.81%, the cell stores the fraction 0.3881
    caveats = check_numeric_grounding("Blackwell share is 38.81%.", _sql([[0.3881]]), [], [])
    assert caveats == []


def test_magnitude_suffix_matches_raw_cell():
    # prose "$81,615M" == 81,615,000,000 in the cell
    caveats = check_numeric_grounding("Revenue $81,615M.", _sql([[81615000000]]), [], [])
    assert caveats == []


def test_years_and_small_counts_are_noise():
    caveats = check_numeric_grounding("In FY2027 we found 3 tables.", [], [], [])
    assert caveats == []


def test_grounded_by_api_data_or_doc_quote():
    api = [{"source": "itsm", "path": "/tickets", "params": {}, "status": 200,
            "row_count": 1, "data": [{"open": 8}]}]
    docs = [{"doc_id": "doc:x", "chunk_id": "doc:x:chunk:1",
             "quote": "Revenue $81,615M", "score": 0.9}]
    assert check_numeric_grounding("There are 8 open tickets.", [], api, []) == []
    assert check_numeric_grounding("Revenue was $81,615M.", [], [], docs) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_grounding.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'semantic_layer.web.grounding'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/semantic_layer/web/grounding.py
"""Deterministic groundedness guard: flag summary numbers absent from results.

A guard, not a proof. It catches fabricated or mis-transcribed figures, not
wrong-but-present ones, and is intentionally generous (wide tolerance + magnitude
and percent variants) to keep false positives low. Caveats are advisory and never
block an answer.
"""

import json
import re
from typing import Any

# A number with optional leading $, optional thousands separators, optional decimal,
# and an optional magnitude/percent suffix.
_NUM = re.compile(r"\$?\s?(\d[\d,]*(?:\.\d+)?)\s?([kKmMbB%])?")
_NOISE_YEARS = {2024.0, 2025.0, 2026.0, 2027.0}


def _variants(value: float, suffix: str) -> set[float]:
    """Expand one written number into the magnitudes it could mean."""
    out = {value}
    s = suffix.lower()
    if s == "k":
        out.add(value * 1e3)
    elif s == "m":
        out.add(value * 1e6)
    elif s == "b":
        out.add(value * 1e9)
    elif s == "%":
        out.add(value / 100.0)   # "38.81%" may be stored as 0.3881
    return out


def _numbers_in_text(text: str) -> set[float]:
    """Every number mention in free text, expanded to its magnitude variants."""
    found: set[float] = set()
    for raw, suffix in _NUM.findall(text or ""):
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        found |= _variants(value, suffix)
    return found


def _cell_numbers(cell: Any) -> set[float]:
    if isinstance(cell, bool):
        return set()
    if isinstance(cell, (int, float)):
        return {float(cell)}
    if isinstance(cell, str):
        return _numbers_in_text(cell)
    return set()


def _grounded_set(sql_runs: list, api_calls: list, doc_citations: list) -> set[float]:
    grounded: set[float] = set()
    for run in sql_runs:
        for row in run.get("rows", []) or []:
            for cell in row:
                grounded |= _cell_numbers(cell)
    blob = json.dumps([c.get("data") for c in api_calls], default=str)
    blob += " " + " ".join(c.get("quote", "") for c in doc_citations)
    grounded |= _numbers_in_text(blob)
    return grounded


def _matches(candidates: set[float], grounded: set[float]) -> bool:
    for a in candidates:
        for b in grounded:
            if abs(a - b) <= max(0.5, 0.01 * max(abs(a), abs(b))):
                return True
    return False


def _is_noise(value: float, suffix: str) -> bool:
    if not suffix and value.is_integer():
        if value in _NOISE_YEARS:      # calendar/fiscal years
            return True
        if abs(value) < 10:            # small ordinals/counts ("3 tables")
            return True
    return False


def check_numeric_grounding(content: str, sql_runs: list, api_calls: list,
                            doc_citations: list) -> list[str]:
    """Return one advisory caveat per summary number not found in any result."""
    grounded = _grounded_set(sql_runs, api_calls, doc_citations)
    caveats: list[str] = []
    seen: set[str] = set()
    for raw, suffix in _NUM.findall(content or ""):
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        if _is_noise(value, suffix):
            continue
        display = f"{raw}{suffix}"
        if display in seen:
            continue
        if not _matches(_variants(value, suffix), grounded):
            seen.add(display)
            caveats.append(
                f"The summary cites {display}, which was not found in any "
                f"retrieved result (SQL/API/document)."
            )
    return caveats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_grounding.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/web/grounding.py backend/tests/test_grounding.py
git commit -m "feat(web): deterministic numeric grounding guard"
```

---

## Task 2: Provenance collector (`_Provenance` in `events.py`)

**Files:**
- Modify: `backend/semantic_layer/web/events.py`
- Test: `backend/tests/test_provenance.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_provenance.py
import json

from semantic_layer.web.events import _Provenance


def test_records_sql_run_with_sql_and_rows():
    prov = _Provenance()
    prov.record("run_sql", {"source": "sales_pg", "sql": "SELECT 1"},
                json.dumps({"columns": ["n"], "rows": [[1], [2]]}))
    fields = prov.answer_fields("ok")
    assert len(fields["sql_runs"]) == 1
    run = fields["sql_runs"][0]
    assert run["source"] == "sales_pg"
    assert run["sql"] == "SELECT 1"
    assert run["columns"] == ["n"]
    assert run["row_count"] == 2
    assert run["error"] is None


def test_records_sql_error():
    prov = _Provenance()
    prov.record("run_sql", {"source": "org", "sql": "SELECT bad"},
                json.dumps({"error": "no such column: bad"}))
    run = prov.answer_fields("ok")["sql_runs"][0]
    assert run["error"] == "no such column: bad"
    assert run["rows"] == []


def test_records_api_call():
    prov = _Provenance()
    prov.record("call_api", {"source": "itsm", "path": "/tickets", "params": {"status": "open"}},
                json.dumps({"status": 200, "data": [{"id": 1}, {"id": 2}]}))
    call = prov.answer_fields("ok")["api_calls"][0]
    assert call["source"] == "itsm"
    assert call["path"] == "/tickets"
    assert call["row_count"] == 2


def test_dedupes_doc_citations_by_chunk():
    prov = _Provenance()
    payload = json.dumps([
        {"chunk_id": "doc:x:chunk:1", "doc_id": "doc:x", "text": "hello", "score": 0.9},
        {"chunk_id": "doc:x:chunk:1", "doc_id": "doc:x", "text": "hello", "score": 0.9},
    ])
    prov.record("search_documents", {"query": "q"}, payload)
    cites = prov.answer_fields("ok")["doc_citations"]
    assert len(cites) == 1
    assert cites[0]["chunk_id"] == "doc:x:chunk:1"
    assert cites[0]["quote"] == "hello"


def test_answer_fields_runs_grounding():
    prov = _Provenance()
    prov.record("run_sql", {"source": "sales_pg", "sql": "SELECT 1"},
                json.dumps({"columns": ["n"], "rows": [[5]]}))
    fields = prov.answer_fields("The number is 99999.")
    assert any("99999" in c for c in fields["caveats"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_provenance.py -v`
Expected: FAIL with `ImportError: cannot import name '_Provenance' from 'semantic_layer.web.events'`

- [ ] **Step 3: Write minimal implementation**

Add the import and the `_Provenance` class to `backend/semantic_layer/web/events.py`. Insert the import near the top (after the existing `from semantic_layer.agent.build import build_agent` on line 6):

```python
from semantic_layer.web.grounding import check_numeric_grounding
```

Then add the class immediately after `_collect_highlight` (after line 26):

```python
class _Provenance:
    """Accumulate structured SQL/API/doc provenance from paired tool results."""

    def __init__(self) -> None:
        self.sql_runs: list[dict] = []
        self.api_calls: list[dict] = []
        self.doc_citations: list[dict] = []
        self._seen_chunks: set[str] = set()

    def record(self, name: str, args: dict, content: str) -> None:
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return
        if name == "run_sql":
            entry = {"source": args.get("source"), "sql": args.get("sql", "")}
            if isinstance(data, dict) and "error" in data:
                entry.update(columns=[], rows=[], row_count=0, error=data["error"])
            elif isinstance(data, dict):
                rows = data.get("rows", []) or []
                entry.update(columns=data.get("columns", []), rows=rows,
                             row_count=len(rows), error=None)
            else:
                return
            self.sql_runs.append(entry)
        elif name == "call_api":
            if not isinstance(data, dict):
                return
            body = data.get("data")
            row_count = len(body) if isinstance(body, list) else (1 if body else 0)
            self.api_calls.append({
                "source": args.get("source"), "path": args.get("path"),
                "params": args.get("params") or {}, "status": data.get("status"),
                "row_count": row_count, "data": body,
            })
        elif name == "search_documents":
            if not isinstance(data, list):
                return
            for hit in data:
                cid = hit.get("chunk_id")
                if not cid or cid in self._seen_chunks:
                    continue
                self._seen_chunks.add(cid)
                self.doc_citations.append({
                    "doc_id": hit.get("doc_id"), "chunk_id": cid,
                    "quote": (hit.get("text") or "")[:280], "score": hit.get("score"),
                })

    def answer_fields(self, content: str) -> dict:
        return {
            "sql_runs": self.sql_runs,
            "api_calls": self.api_calls,
            "doc_citations": self.doc_citations,
            "caveats": check_numeric_grounding(
                content, self.sql_runs, self.api_calls, self.doc_citations),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_provenance.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/web/events.py backend/tests/test_provenance.py
git commit -m "feat(web): structured provenance collector for tool results"
```

---

## Task 3: Wire provenance into the stream + enrich the `answer` event

**Files:**
- Modify: `backend/semantic_layer/web/events.py:29-54` (`stream_chat_events`)
- Test: `backend/tests/test_web_chat_events.py`

- [ ] **Step 1: Write the failing test (extend the existing end-to-end test)**

Append to `backend/tests/test_web_chat_events.py`:

```python
@pytest.mark.neo4j
@pytest.mark.postgres
@pytest.mark.openai
def test_answer_event_carries_sql_provenance(ingested_graph, require_openai):
    events = list(stream_chat_events(
        "Which segment has the most revenue? Use the sales database."))
    answer = events[-1]
    assert answer["type"] == "answer"
    # New structured fields exist and the SQL run was captured with its query text.
    assert isinstance(answer["sql_runs"], list) and answer["sql_runs"]
    run = answer["sql_runs"][0]
    assert run["sql"].strip().lower().startswith(("select", "with"))
    assert run["columns"]
    assert run["row_count"] >= 1
    assert "caveats" in answer and isinstance(answer["caveats"], list)
    assert "doc_citations" in answer and "api_calls" in answer
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_web_chat_events.py::test_answer_event_carries_sql_provenance -v`
Expected: FAIL with `KeyError: 'sql_runs'`

(If Neo4j/Postgres/OpenAI markers cause a skip locally, run the full `setup.sh` services first, then `make ingest`.)

- [ ] **Step 3: Replace `stream_chat_events` body to pair calls↔results and enrich the answer**

Replace `backend/semantic_layer/web/events.py:29-54` with:

```python
def stream_chat_events(question: str) -> Iterator[dict]:
    """Yield UI events: {type: tool_call|tool_result|answer, ...}."""
    agent = build_agent()
    highlight: set[str] = set()
    prov = _Provenance()
    pending: dict[str, dict] = {}   # tool_call_id -> args, to pair with its result
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
                    if call.get("id"):
                        pending[call["id"]] = call.get("args", {})
                    yield {"type": "tool_call", "scope": scope,
                           "name": call.get("name"), "args": call.get("args", {})}
                if type(m).__name__ == "ToolMessage":
                    name = getattr(m, "name", "")
                    content = str(getattr(m, "content", ""))
                    args = pending.get(getattr(m, "tool_call_id", None), {})
                    _collect_highlight(name, content, highlight)
                    prov.record(name, args, content)
                    yield {"type": "tool_result", "scope": scope,
                           "name": name, "content": content[:4000]}
                elif type(m).__name__ == "AIMessage" and getattr(m, "content", None) \
                        and not getattr(m, "tool_calls", None):
                    final = m.content
    yield {"type": "answer", "content": final, "highlight": sorted(highlight),
           **prov.answer_fields(final)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_web_chat_events.py -v`
Expected: PASS (both the original `test_stream_emits_tools_and_final_answer` and the new test)

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/web/events.py backend/tests/test_web_chat_events.py
git commit -m "feat(web): emit sql/api/doc provenance + caveats on the answer event"
```

---

## Task 4: Extend the frontend `ChatEvent` types

**Files:**
- Modify: `frontend/lib/types.ts`

- [ ] **Step 1: Add the new types**

Append to `frontend/lib/types.ts` and replace the `answer` variant of `ChatEvent`:

```typescript
export type SqlRun = {
  source: string;
  sql: string;
  columns: string[];
  rows: (string | number | null)[][];
  row_count: number;
  error: string | null;
};
export type ApiCall = {
  source: string;
  path: string;
  params: Record<string, unknown>;
  status: number | null;
  row_count: number;
  data: unknown;
};
export type DocCitation = {
  doc_id: string;
  chunk_id: string;
  quote: string;
  score: number | null;
};
```

Then change the `answer` member of the `ChatEvent` union from:

```typescript
  | { type: "answer"; content: string; highlight: string[] };
```

to:

```typescript
  | {
      type: "answer";
      content: string;
      highlight: string[];
      sql_runs: SqlRun[];
      api_calls: ApiCall[];
      doc_citations: DocCitation[];
      caveats: string[];
    };

export type AnswerEvent = Extract<ChatEvent, { type: "answer" }>;
```

- [ ] **Step 2: Type-check**

Run: `cd frontend && npm run build`
Expected: compiles past type-checking (it will fail later in this build only if `ChatPanel`/`useChatStream` still reference old shapes — those are fixed in Tasks 5–6; if you run the build now expect the error to point at `useChatStream.ts`, which is fine).

- [ ] **Step 3: Commit**

```bash
git add frontend/lib/types.ts
git commit -m "feat(web-ui): add SqlRun/ApiCall/DocCitation answer-event types"
```

---

## Task 5: Expose the full answer event from `useChatStream`

**Files:**
- Modify: `frontend/lib/useChatStream.ts`

- [ ] **Step 1: Track the answer event**

In `frontend/lib/useChatStream.ts`, import the new type and add state. Change the import line:

```typescript
import type { ChatEvent } from "./types";
```

to:

```typescript
import type { ChatEvent, AnswerEvent } from "./types";
```

Add state alongside the others:

```typescript
  const [answerEvent, setAnswerEvent] = useState<AnswerEvent | null>(null);
```

In `reset`, clear it inside the existing `if (!b)` block:

```typescript
        setEvents([]);
        setAnswer("");
        setHighlight([]);
        setAnswerEvent(null);
```

In `ask`, clear it at the start (next to `setAnswer("")`):

```typescript
    setAnswerEvent(null);
```

In the SSE loop, when an answer event arrives, capture the whole event:

```typescript
          if (evt.type === "answer") {
            setAnswer(evt.content);
            setHighlight(evt.highlight);
            setAnswerEvent(evt);
          }
```

Return it:

```typescript
  return { events, answer, answerEvent, highlight, busy, ask, reset };
```

- [ ] **Step 2: Type-check**

Run: `cd frontend && npm run build`
Expected: compiles past `useChatStream.ts` (may still error in `page.tsx`/`ChatPanel.tsx` until Task 6).

- [ ] **Step 3: Commit**

```bash
git add frontend/lib/useChatStream.ts
git commit -m "feat(web-ui): expose full answer event from useChatStream"
```

---

## Task 6: AnswerPanel + wire into ChatPanel and page

**Files:**
- Create: `frontend/components/AnswerPanel.tsx`
- Modify: `frontend/components/ChatPanel.tsx`
- Modify: `frontend/app/page.tsx`

- [ ] **Step 1: Create `AnswerPanel.tsx`**

```tsx
// frontend/components/AnswerPanel.tsx
"use client";
import type { AnswerEvent, SqlRun } from "@/lib/types";

function ResultTable({ run }: { run: SqlRun }) {
  if (run.error)
    return <div className="text-xs text-red-400 font-mono">error: {run.error}</div>;
  const rows = run.rows.slice(0, 50);
  return (
    <div className="overflow-auto max-h-64 border border-gray-800 rounded">
      <table className="text-xs w-full">
        <thead className="bg-gray-900 text-gray-400 sticky top-0">
          <tr>
            {run.columns.map((c) => (
              <th key={c} className="text-left px-2 py-1 font-medium">{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-t border-gray-800">
              {row.map((cell, j) => (
                <td key={j} className="px-2 py-1 text-gray-200">{String(cell)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {run.row_count > rows.length && (
        <div className="text-[10px] text-gray-500 px-2 py-1">
          showing {rows.length} of {run.row_count} rows
        </div>
      )}
    </div>
  );
}

export function AnswerPanel({
  answer,
  onSelectNode,
}: {
  answer: AnswerEvent;
  onSelectNode?: (id: string) => void;
}) {
  return (
    <div className="space-y-3">
      <div className="rounded-lg bg-gray-800 p-3 text-gray-100 whitespace-pre-wrap">
        {answer.content}
      </div>

      {answer.caveats.length > 0 && (
        <div className="rounded border border-amber-700 bg-amber-950/40 p-2 text-xs text-amber-300 space-y-1">
          <div className="font-medium">Groundedness check</div>
          {answer.caveats.map((c, i) => (
            <div key={i}>⚠ {c}</div>
          ))}
        </div>
      )}

      {answer.sql_runs.map((run, i) => (
        <details key={i} className="rounded border border-gray-800" open={i === 0}>
          <summary className="cursor-pointer px-2 py-1 text-xs text-blue-300">
            Generated SQL · <span className="text-gray-500">{run.source}</span>
          </summary>
          <pre className="text-[11px] font-mono text-gray-300 px-2 py-1 overflow-auto whitespace-pre-wrap">
            {run.sql}
          </pre>
          <div className="px-2 pb-2">
            <ResultTable run={run} />
          </div>
        </details>
      ))}

      {answer.api_calls.map((call, i) => (
        <div key={i} className="rounded border border-gray-800 px-2 py-1 text-xs text-gray-300">
          API · <span className="text-blue-300">{call.source}{call.path}</span>{" "}
          <span className="text-gray-500">({call.row_count} rows)</span>
        </div>
      ))}

      {answer.doc_citations.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {answer.doc_citations.map((c) => (
            <button
              key={c.chunk_id}
              title={c.quote}
              onClick={() => onSelectNode?.(c.chunk_id)}
              className="text-[11px] rounded border border-gray-700 px-2 py-1 text-gray-300 hover:border-[#76b900]"
            >
              {c.doc_id}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Wire `AnswerPanel` into `ChatPanel.tsx`**

In `frontend/components/ChatPanel.tsx`, change the imports and props. Add at top:

```tsx
import type { AnswerEvent } from "@/lib/types";
import { AnswerPanel } from "./AnswerPanel";
```

Change the component signature from `answer: string` to the answer event:

```tsx
export function ChatPanel({
  answerEvent,
  busy,
  onAsk,
  onReset,
}: {
  answerEvent: AnswerEvent | null;
  busy: boolean;
  onAsk: (q: string) => void;
  onReset: () => void;
}) {
```

Replace every `answer` reference in the body. The `onReset` button guard (`:27`) and the examples guard (`:35`) become `answerEvent`:

```tsx
        {answerEvent && !busy && (
          <button
            onClick={onReset}
            className="text-sm text-gray-400 hover:text-[#76b900]"
          >
            ← Back to questions
          </button>
        )}
        {!answerEvent && !busy && (
```

Replace the plain answer block (`:49-51`) with:

```tsx
        {answerEvent && <AnswerPanel answer={answerEvent} />}
```

- [ ] **Step 3: Update `page.tsx` to pass `answerEvent`**

In `frontend/app/page.tsx`, change the hook destructure to include `answerEvent` (drop the now-unused `answer` if your linter complains):

```tsx
  const { events, answerEvent, highlight, busy, ask, reset } = useChatStream();
```

and the `ChatPanel` usage:

```tsx
          <ChatPanel answerEvent={answerEvent} busy={busy} onAsk={ask} onReset={reset} />
```

- [ ] **Step 4: Build to verify the whole frontend type-checks**

Run: `cd frontend && npm run build`
Expected: build succeeds with no type errors.

- [ ] **Step 5: Manual verification (no frontend test runner exists)**

1. `./start-backend.sh` then `./start-ui.sh` (UI on http://localhost:3005).
2. Ask the example: *"Which business segment has the highest total revenue?"*
   - Confirm: NL summary, a collapsible **Generated SQL** block with the query, and a **results table** with `segment`/revenue columns.
3. Ask: *"How many open support tickets are there?"* → confirm an **API ·** line appears.
4. Ask a press-release question → confirm a **doc citation chip**; clicking it does nothing harmful (node-selection wiring is optional and covered in a follow-up).
5. Sanity-check a caveat: not generally reproducible on demand — verified by the backend unit tests in Task 1.

- [ ] **Step 6: Commit**

```bash
git add frontend/components/AnswerPanel.tsx frontend/components/ChatPanel.tsx frontend/app/page.tsx
git commit -m "feat(web-ui): consumption-layer answer panel (SQL + results + citations + caveats)"
```

---

## Self-Review notes

- **Spec coverage:** payload shape (Task 3) ✓; provenance capture/pairing (Tasks 2–3) ✓; `grounding.py` guard (Task 1) ✓; `lib/types.ts` (Task 4) ✓; `AnswerPanel` with SQL+table+citations+caveats (Task 6) ✓; `ChatPanel` swap (Task 6) ✓; `useChatStream` exposes the event (Task 5) ✓.
- **Deviation from spec:** spec proposed accumulation "in events.py" generically; this plan makes it a unit-testable `_Provenance` class (still in events.py) so it can be tested without a live agent. Grounding signature is `(content, sql_runs, api_calls, doc_citations)` (added `doc_citations`) so doc-/API-only answers don't false-positive.
- **Node-inspector click-through** from citation chips is stubbed (`onSelectNode` optional) — wiring `chunk_id` → `NodeDetails` selection in `page.tsx` is a small follow-up, deliberately out of this plan to keep tasks bite-sized.
- **Type consistency:** `_Provenance.answer_fields()` keys (`sql_runs/api_calls/doc_citations/caveats`) match `AnswerEvent` and `AnswerPanel` usage.
- **Reminder:** running backend pytest wipes chunk embeddings — run `make ingest` before the marked end-to-end test in Task 3 and before manual doc-citation checks.
```
