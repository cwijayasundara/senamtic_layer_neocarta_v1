"""Leg workers: execute one pre-resolved plan slice with a single structured LLM call.

Each leg receives an exact slice of the Plan (tables, joins, filters, endpoints) — no
discovery — writes/selects what to run, executes a deterministic tool, and returns a
structured result the controller folds into the answer.
"""

import json

from pydantic import BaseModel

from semantic_layer.agent.graph_tools import _sql_reference
from semantic_layer.agent.sql_tools import _run
from semantic_layer.agent.api_tools import call_api
from semantic_layer.agent.doc_tools import search_documents
from semantic_layer.ingest.llm import get_chat_model
from semantic_layer.config import settings


class _SqlDraft(BaseModel):
    sql: str


_SQL_LEG_PROMPT = (
    "You are a SQL expert. Write ONE read-only SELECT for the given plan slice and nothing "
    "else. Use the exact table references and join column pairs provided. Match dimension "
    "name filters case-insensitively with ILIKE '%value%'. If a fiscal scope is given and "
    "the fact is order_line, reach the period via order_line->sales_order(order_id)->"
    "fiscal_period(fiscal_period_id) and filter fiscal_year/quarter; if the fact is "
    "income_statement, filter its own fiscal_year/quarter columns. Return only the SQL."
)


def _col(cid: str) -> str:
    """Readable table.column from a column id col:src.schema.table.column."""
    parts = cid.split(":", 1)[1].split(".")
    return f"{parts[-2]}.{parts[-1]}"


def _sql_brief(leg: dict) -> str:
    lines = [f"Source: {leg['source']}",
             f"Fact table: {_sql_reference(leg['fact_table'])}"]
    for jt in leg.get("join_targets", []):
        pairs = ", ".join(f"{_col(j['on'][0])} = {_col(j['on'][1])}" for j in jt["joins"])
        lines.append(f"Join {_sql_reference(jt['table_id'])} ON {pairs}")
    for f in leg.get("filters", []):
        lines.append(f"Filter {_sql_reference(f['table_id'])}.{f['column']} ~ '{f['value']}'")
    scope = leg.get("scope") or {}
    if scope.get("fiscal_year"):
        lines.append(f"Scope: fiscal_year={scope['fiscal_year']} quarter={scope.get('quarter')}")
    if leg.get("metrics"):
        lines.append(f"Select these measures: {', '.join(leg['metrics'])}")
    return "\n".join(lines)


def _draft_and_run(model, brief: str, source: str, extra: str = "") -> tuple[str, dict]:
    draft = model.invoke([("system", _SQL_LEG_PROMPT), ("human", brief + extra)])
    return draft.sql, json.loads(_run(source, draft.sql))


def run_sql_leg(leg: dict) -> dict:
    brief = _sql_brief(leg)
    model = get_chat_model(settings.llm_model).with_structured_output(_SqlDraft)
    sql, out = _draft_and_run(model, brief, leg["source"])
    if isinstance(out, dict) and "error" in out:
        sql, out = _draft_and_run(model, brief, leg["source"],
                                  extra=f"\n\nThe previous SQL failed: {out['error']}\nFix it.")
    if isinstance(out, dict) and "error" in out:
        return {"source": leg["source"], "sql": sql, "columns": [], "rows": [],
                "row_count": 0, "error": out["error"]}
    rows = out.get("rows", [])
    return {"source": leg["source"], "sql": sql, "columns": out.get("columns", []),
            "rows": rows, "row_count": len(rows), "error": None}


# --- API leg -----------------------------------------------------------------

class _ApiCall(BaseModel):
    source: str
    path: str
    params: dict = {}


class _ApiCalls(BaseModel):
    calls: list[_ApiCall] = []


_API_LEG_PROMPT = (
    "Pick the mock-API calls that answer the given lookups. Sources & endpoints: "
    "crm (/accounts,/contacts,/opportunities), itsm (/tickets,/rma), "
    "partner (/partners,/inventory), dgx (/usage). Use query params to filter "
    "(e.g. {'status':'open'} for open tickets). Return the list of calls to make."
)


def run_api_leg(api_intents: list[str]) -> dict:
    model = get_chat_model(settings.llm_model).with_structured_output(_ApiCalls)
    plan = model.invoke([("system", _API_LEG_PROMPT),
                         ("human", "Lookups: " + "; ".join(api_intents))])
    results = []
    for c in plan.calls:
        resp = json.loads(call_api(c.source, c.path, c.params))
        body = resp.get("data")
        row_count = len(body) if isinstance(body, list) else (1 if body else 0)
        results.append({"source": c.source, "path": c.path, "params": c.params,
                        "status": resp.get("status"), "row_count": row_count, "data": body})
    return {"calls": results, "error": None}


# --- Doc leg -----------------------------------------------------------------

class _DocAnswer(BaseModel):
    answer: str


_DOC_LEG_PROMPT = (
    "Answer the question ONLY from the provided document passages, quoting the most "
    "relevant sentence and citing the document id. If nothing relevant is present, say so."
)


def run_doc_leg(doc_query: str) -> dict:
    hits = json.loads(search_documents.invoke({"query": doc_query}))
    citations, doc_texts, seen = [], [], set()
    for h in hits:
        cid = h.get("chunk_id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        text = h.get("text") or ""
        doc_texts.append(text)
        citations.append({"doc_id": h.get("doc_id"), "chunk_id": cid,
                          "quote": text[:280], "score": h.get("score")})
    passages = "\n\n".join(f"[{c['doc_id']}] {t}" for c, t in zip(citations, doc_texts))
    model = get_chat_model(settings.llm_model).with_structured_output(_DocAnswer)
    ans = model.invoke([("system", _DOC_LEG_PROMPT),
                        ("human", f"Question: {doc_query}\n\nPassages:\n{passages}")])
    return {"answer": ans.answer, "citations": citations, "doc_texts": doc_texts, "error": None}
