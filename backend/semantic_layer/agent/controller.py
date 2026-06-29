"""Deterministic controller: extract_intent -> build_plan -> parallel legs -> synthesize.

Emits the same UI event stream as the legacy path (tool_call/tool_result/answer) so the
web UI is unchanged. Bounded LLM calls: extract(1) + legs(<=1 each) + synthesize(1).
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

from semantic_layer.agent.planner import extract_intent, build_plan
from semantic_layer.agent.legs import run_sql_leg, run_api_leg, run_doc_leg
from semantic_layer.ingest.llm import get_chat_model
from semantic_layer.config import settings
from semantic_layer.web.grounding import check_numeric_grounding
from semantic_layer.agent.cache import query_cache, embed_query

_answer_gate = threading.BoundedSemaphore(settings.max_concurrent_answers)

_SYNTH_PROMPT = (
    "Synthesize a concise answer from the leg results below. State which source(s) each "
    "fact came from and cite document ids. Only use numbers present in the results. "
    "If the API results need to be tied to specific customers, correlate via the given keys.\n"
    "EMPTY COHORT: when a SQL leg returns row_count 0, treat it as a real, valid finding — "
    "the requested cohort has no matching records — not an error or missing data. Say so "
    "explicitly and name the exact filters/scope that defined the cohort (read them from the "
    "leg's WHERE clause), e.g. 'no <region> <industry> customers had <product> revenue in "
    "<period>'. Because the cohort is empty there is nothing to correlate to: do NOT report "
    "API or other-leg totals (usage rows, ticket counts, etc.) as if they describe this "
    "cohort. You may state such a total exists overall, but make clear it cannot be "
    "attributed to the empty cohort. Where it helps the reader, note that the empty result "
    "comes from the intersection of the applied filters and suggest relaxing one dimension "
    "(e.g. broaden the period or remove the industry/region constraint) to find rows — but "
    "never invent the rows a relaxed query would return.\n"
    "DIFFERENT-SCOPE FIGURES: do NOT present two numbers as a comparison ('vs.', 'compared "
    "to', up/down deltas) unless they measure the same thing for the same entity over the "
    "same period. In particular, the financials DB revenue is total-company revenue across "
    "all segments, whereas a press-release figure may be a single-segment number (e.g. Data "
    "Center only) and may describe a different reporting entity. Report each such figure on "
    "its own with its scope and source stated; if they are not like-for-like, say so plainly "
    "('not a like-for-like comparison: $X is total-company revenue, $Y is Data-Center-only "
    "from the press release') instead of implying one beats or trails the other. A sanity "
    "check: a single segment cannot exceed total company revenue — if it appears to, the two "
    "numbers are from different entities and must not be compared."
)


def _table_id_from_column_id(column_id: str | None) -> str | None:
    if not isinstance(column_id, str) or not column_id.startswith("col:"):
        return None
    prefix, _, _column = column_id.rpartition(".")
    if not prefix:
        return None
    return "table:" + prefix.split(":", 1)[1]


def _api_table_id(call: dict) -> str | None:
    source = call.get("source")
    path = call.get("path")
    if not isinstance(source, str) or not isinstance(path, str):
        return None
    return f"table:{source}.api.GET {path}"


def _answer_highlight(plan: dict, api_calls: list[dict], doc_citations: list[dict]) -> list[str]:
    out = set(plan.get("highlight", []))
    for leg in plan.get("sql_legs", []):
        fact_table = leg.get("fact_table")
        if isinstance(fact_table, str):
            out.add(fact_table)
        for target in leg.get("join_targets", []):
            table_id = target.get("table_id")
            if isinstance(table_id, str):
                out.add(table_id)
            out.update(t for t in target.get("tables", []) if isinstance(t, str))
    for corr in plan.get("api_correlations", []):
        for key in ("sql_column", "api_column"):
            table_id = _table_id_from_column_id(corr.get(key))
            if table_id:
                out.add(table_id)
    for call in api_calls:
        table_id = _api_table_id(call)
        if table_id:
            out.add(table_id)
    for citation in doc_citations:
        for key in ("doc_id", "chunk_id"):
            node_id = citation.get(key)
            if isinstance(node_id, str):
                out.add(node_id)
    return sorted(out)


def _synthesize(question: str, sql_runs, api_calls, doc, correlations) -> str:
    payload = {"question": question, "sql": sql_runs, "api": api_calls,
               "doc_answer": (doc or {}).get("answer"), "api_correlations": correlations}
    model = get_chat_model(settings.synthesis_model_resolved)
    msg = model.invoke([("system", _SYNTH_PROMPT), ("human", json.dumps(payload, default=str))])
    return msg.content


def answer_stream(question: str) -> Iterator[dict]:
    q_embedding = None
    if settings.query_cache_enabled:
        cached = query_cache.get_exact(question)
        if cached is None:
            try:
                q_embedding = embed_query(question)
                cached = query_cache.get_semantic(q_embedding, settings.cache_similarity_threshold)
            except Exception:  # noqa: BLE001 — a cache miss must never block answering
                q_embedding = None
        if cached is not None:
            # Replay the full stored event list; mark the answer event as cached.
            for ev in cached:
                yield ({**ev, "cached": True} if ev.get("type") == "answer" else ev)
            return

    # Collect every success-path event so we can store the full stream in the cache.
    collected: list[dict] = []

    def _emit(ev: dict) -> dict:
        collected.append(ev)
        return ev

    with _answer_gate:
        try:
            intent = extract_intent(question)
            plan = build_plan(intent, question=question)
            yield _emit({"type": "tool_result", "scope": "plan", "name": "plan_query",
                         "content": json.dumps({k: plan[k] for k in ("highlight",) if k in plan})[:4000]})

            # Fan out independent legs concurrently.
            jobs = {}
            with ThreadPoolExecutor(max_workers=settings.leg_max_workers) as pool:
                for leg in plan.get("sql_legs", []):
                    jobs[pool.submit(run_sql_leg, leg)] = ("sql", leg["source"])
                if intent.needs_api and intent.api_intents:
                    jobs[pool.submit(run_api_leg, intent.api_intents)] = ("api", "api")
                if plan.get("doc_leg"):
                    jobs[pool.submit(run_doc_leg, plan["doc_leg"]["doc_query"])] = ("doc", "doc")

                sql_runs, api_calls, doc_texts, doc_citations, doc = [], [], [], [], None
                trace = []
                for fut in list(jobs):
                    kind, label = jobs[fut]
                    _t0 = time.perf_counter()
                    try:
                        res = fut.result()
                    except Exception as exc:  # noqa: BLE001 — one leg failing must not sink the answer
                        if kind == "sql":
                            res = {"source": label, "sql": "", "columns": [], "rows": [],
                                   "row_count": 0, "error": str(exc)}
                        elif kind == "api":
                            res = {"calls": [], "error": str(exc)}
                        else:
                            res = {"answer": "", "citations": [], "doc_texts": [], "error": str(exc)}
                    yield _emit({"type": "tool_result", "scope": kind, "name": f"{kind}_leg",
                                 "content": json.dumps(res, default=str)[:4000]})
                    if kind == "sql":
                        sql_runs.append(res)
                    elif kind == "api":
                        api_calls.extend(res.get("calls", []))
                    elif kind == "doc":
                        doc = res
                        doc_citations = res.get("citations", [])
                        doc_texts = res.get("doc_texts", [])
                    trace.append({"name": f"{kind}:{label}",
                                  "duration_ms": round((time.perf_counter() - _t0) * 1000, 2),
                                  "ok": (res.get("error") is None)})

            summary = _synthesize(question, sql_runs, api_calls, doc,
                                  plan.get("api_correlations", []))
            caveats = check_numeric_grounding(summary, sql_runs, api_calls, doc_texts)
        except Exception as exc:  # noqa: BLE001 — never leave the UI hanging
            # Exception path: yield but do NOT collect/cache — partial runs must not be stored.
            yield {"type": "answer", "content": f"The agent could not complete this question: {exc}",
                   "highlight": [], "sql_runs": [], "api_calls": [], "doc_citations": [], "caveats": []}
            return

        answer_event = {"type": "answer", "content": summary,
                        "highlight": _answer_highlight(plan, api_calls, doc_citations),
                        "sql_runs": sql_runs, "api_calls": api_calls,
                        "doc_citations": doc_citations, "caveats": caveats, "trace": trace}
        # Store the full event list (tool_results + answer) so replays include the reasoning trace.
        if settings.query_cache_enabled:
            _emit(answer_event)  # add answer_event to collected before storing
            query_cache.put(question, collected, embedding=q_embedding)
            yield answer_event
        else:
            yield answer_event
