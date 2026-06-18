"""Schema routing: retrieve a high-recall set of candidate tables for a question,
then rank them with one LLM call. Decouples table selection from SQL generation
so text-to-SQL scales past a handful of tables (cf. DBCopilot, arXiv 2312.03463;
LinkedIn text-to-SQL, arXiv 2507.14372)."""

import json

from semantic_layer.agent.graph_tools import search_catalog


def retrieve_candidate_tables(question: str, k_ret: int = 20) -> list[dict]:
    """High-recall candidate tables for a question.

    Aggregates search_catalog hits (keyword/value/business-term) by their owning
    table, summing hit scores. Returns [{table_id, score}] ranked DESC, capped at
    k_ret. Tuned for recall: k_ret is intentionally generous; the LLM ranker
    (rank_tables) trims to a precise set."""
    hits = json.loads(search_catalog.invoke({"query": question}))
    scores: dict[str, int] = {}
    for h in hits:
        tid = h.get("table_id")
        if not tid or not tid.startswith("table:"):
            continue
        scores[tid] = scores.get(tid, 0) + int(h.get("score") or 1)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"table_id": tid, "score": s} for tid, s in ranked[:k_ret]]
