"""Translate the agent's streamed steps into UI events + a highlight node-set."""

import json
from typing import Iterator

from semantic_layer.web.grounding import check_numeric_grounding


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


class _Provenance:
    """Accumulate structured SQL/API/doc provenance from paired tool results."""

    def __init__(self) -> None:
        self.sql_runs: list[dict] = []
        self.api_calls: list[dict] = []
        self.doc_citations: list[dict] = []
        self._doc_texts: list[str] = []   # full chunk text for grounding (not truncated)
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
                text = hit.get("text") or ""
                self._doc_texts.append(text)
                self.doc_citations.append({
                    "doc_id": hit.get("doc_id"), "chunk_id": cid,
                    "quote": text[:280], "score": hit.get("score"),
                })

    def answer_fields(self, content: str) -> dict:
        return {
            "sql_runs": self.sql_runs,
            "api_calls": self.api_calls,
            "doc_citations": self.doc_citations,
            "caveats": check_numeric_grounding(
                content, self.sql_runs, self.api_calls, self._doc_texts),
        }


def stream_chat_events(question: str) -> Iterator[dict]:
    """Yield UI events by driving the graph-native controller.

    The controller (extract_intent -> build_plan -> parallel legs -> synthesize) emits the
    same {type: tool_result|answer, ...} events the UI consumes, with bounded LLM calls.
    """
    from semantic_layer.agent.controller import answer_stream
    yield from answer_stream(question)
