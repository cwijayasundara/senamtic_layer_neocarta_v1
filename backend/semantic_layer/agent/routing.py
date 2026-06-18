"""Schema routing: retrieve a high-recall set of candidate tables for a question,
then rank them with one LLM call. Decouples table selection from SQL generation
so text-to-SQL scales past a handful of tables (cf. DBCopilot, arXiv 2312.03463;
LinkedIn text-to-SQL, arXiv 2507.14372)."""

import json

from pydantic import BaseModel, Field

from semantic_layer.agent.driver import driver
from semantic_layer.agent.graph_tools import search_catalog
from semantic_layer.config import settings
from semantic_layer.ingest.llm import get_chat_model


class _TableScore(BaseModel):
    table_id: str
    score: int = Field(ge=0, le=5)


class _TableScores(BaseModel):
    scores: list[_TableScore] = Field(default_factory=list)


_RANK_PROMPT = (
    "You route a business question to the database tables needed to answer it. "
    "For EACH candidate table id, score 0-5 how likely it is required: 5 = certainly "
    "needed (holds the measure or a filter dimension), 0 = irrelevant. Only score the "
    "candidates given; do not invent table ids. Return the list of {table_id, score}."
)


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


def rank_tables(question: str, candidates: list[dict], k_rank: int = 8,
                min_score: int = 3) -> list[str]:
    """One structured LLM call scoring each candidate table's relevance; returns the
    top k_rank table ids scoring >= min_score, ranked by score DESC."""
    if not candidates:
        return []
    model = get_chat_model(settings.planner_model_resolved).with_structured_output(_TableScores)
    listing = "\n".join(f"- {c['table_id']}" for c in candidates)
    result = model.invoke([
        ("system", _RANK_PROMPT),
        ("human", f"Question: {question}\n\nCandidate tables:\n{listing}"),
    ])
    kept = sorted(
        (s for s in result.scores if s.score >= min_score),
        key=lambda s: (-s.score, s.table_id),
    )
    return [s.table_id for s in kept[:k_rank]]


def route_tables(question: str, k_ret: int = 20, k_rank: int = 8) -> list[str]:
    """Retrieve high-recall candidates, then LLM-rank to a precise ordered set."""
    candidates = retrieve_candidate_tables(question, k_ret=k_ret)
    return rank_tables(question, candidates, k_rank=k_rank)


_FACT_RANK_CYPHER = """
UNWIND $tables AS tid
MATCH (t:Table {id: tid})
WHERE tid STARTS WITH 'table:sales_pg.sales.'
OPTIONAL MATCH (t)-[:HAS_COLUMN]->(:Column)-[:REFERENCES]->(ref:Column)
WITH t, tid, count(ref) AS fks
OPTIONAL MATCH (t)-[:HAS_COLUMN]->(:Column)-[:REFERENCES]->(:Column)<-[:HAS_COLUMN]-(:Table)-[:HAS_COLUMN]->(:Column)-[:REFERENCES]->(:Column)<-[:HAS_COLUMN]-(t2:Table)
WHERE t2 <> t
WITH tid, fks, count(DISTINCT t2) AS depth2
RETURN tid ORDER BY fks DESC, depth2 DESC, tid LIMIT 1
"""


def select_fact_table(routed_tables: list[str]) -> str | None:
    """Pick the SQL fact table from a routed set: the sales-schema table with the
    most direct foreign keys, tie-broken by reach at depth 2 (distinct tables
    reachable via two FK hops). Returns None when none qualify."""
    sales = [t for t in routed_tables if t.startswith("table:sales_pg.sales.")]
    if not sales:
        return None
    recs = driver().execute_query(
        _FACT_RANK_CYPHER, tables=sales, database_=settings.neo4j_database,
    ).records
    return recs[0]["tid"] if recs else None
