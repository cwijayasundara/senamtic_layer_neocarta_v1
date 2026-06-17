"""Graph-native planner: one LLM intent pass, then deterministic graph planning.

extract_intent(question) -> Intent     (one structured LLM call, planner_model)
build_plan(intent)       -> Plan dict   (pure Cypher; added in a later task)

This replaces the orchestrator's ~20 LLM discovery round-trips with a single intent
read plus a few set-based graph queries.
"""

import json

from pydantic import BaseModel, Field

from semantic_layer.agent.driver import driver
from semantic_layer.agent.graph_tools import get_join_path
from semantic_layer.config import settings
from semantic_layer.ingest.llm import get_chat_model
from semantic_layer.ingest.value_indexer import norm

_INTENT_PROMPT = (
    "You read a business question over an NVIDIA enterprise semantic layer that unifies "
    "SQL databases, REST APIs, and documents. Extract a structured intent.\n"
    "- terms: the dimension FILTER descriptors mentioned (e.g. 'EMEA','Cloud','Blackwell',"
    "'Data Center'). Split compound noun phrases into separate descriptors.\n"
    "- fact: the measure/metric in plain words (e.g. 'revenue','gpu usage','open tickets'), or null.\n"
    "- group_by: dimensions to break results down by (e.g. ['customer','quarter']).\n"
    "- fiscal_year / quarter: a fiscal scope if stated (e.g. 2025 / 'Q1'), else null.\n"
    "- needs_sql / needs_api / needs_doc: which source TYPES the question requires.\n"
    "- doc_query: what to look up in the documents, or null.\n"
    "- api_intents: enterprise-system lookups implied (e.g. ['dgx usage','open tickets'])."
    "\n- financial_metrics: company-wide financial measures asked for "
    "(e.g. ['revenue','gross margin','net income']); empty unless the question asks "
    "about overall company financials."
)


class Intent(BaseModel):
    terms: list[str] = Field(default_factory=list)
    fact: str | None = None
    group_by: list[str] = Field(default_factory=list)
    fiscal_year: int | None = None
    quarter: str | None = None
    needs_sql: bool = True
    needs_api: bool = False
    needs_doc: bool = False
    doc_query: str | None = None
    api_intents: list[str] = Field(default_factory=list)
    financial_metrics: list[str] = Field(default_factory=list)


def extract_intent(question: str) -> Intent:
    """One structured LLM call (planner_model) -> Intent."""
    model = get_chat_model(settings.planner_model_resolved).with_structured_output(Intent)
    return model.invoke([("system", _INTENT_PROMPT), ("human", question)])


_SALES_FACT = "table:sales_pg.sales.order_line"

_RESOLVE_CYPHER = """
UNWIND $rows AS row
MATCH (db:Database)-[:HAS_SCHEMA]->(:Schema)-[:HAS_TABLE]->(t:Table)
     -[:HAS_COLUMN]->(c:Column)-[:HAS_VALUE]->(v:Value)
WHERE v.norm = row.norm OR v.norm CONTAINS row.norm
RETURN row.term AS term, db.name AS source, t.id AS table_id,
       c.name AS column, v.name AS exact
"""


def _resolve_values(terms: list[str]) -> list[dict]:
    rows = [{"term": t, "norm": norm(t)} for t in terms]
    recs = driver().execute_query(
        _RESOLVE_CYPHER, rows=rows, database_=settings.neo4j_database,
    ).records
    return [dict(r) for r in recs]


def _join_targets(fact: str, table_ids: list[str]) -> list[dict]:
    out = []
    for tid in dict.fromkeys(table_ids):  # de-dupe, preserve order
        path = json.loads(get_join_path.invoke({"table_a_id": fact, "table_b_id": tid}))
        if path.get("found"):
            out.append({"table_id": tid, "tables": path["tables"], "joins": path["joins"]})
    return out


_DOC_CYPHER = """
UNWIND $norms AS n
MATCH (e:Entity)-[:REFERS_TO]->(:Value {norm: n})
MATCH (ch:Chunk)-[:MENTIONS]->(e), (d:Document)-[:HAS_CHUNK]->(ch)
OPTIONAL MATCH (d)-[:COVERS_PERIOD]->(p:Period)
RETURN DISTINCT d.id AS doc_id, collect(DISTINCT p.key) AS periods
"""

_API_CORR_CYPHER = """
MATCH (api:Column)-[:SAME_ENTITY]->(sql:Column)
RETURN sql.id AS sql_column, api.id AS api_column ORDER BY api_column
"""


def _context_docs(terms: list[str]) -> dict | None:
    norms = [norm(t) for t in terms]
    recs = driver().execute_query(
        _DOC_CYPHER, norms=norms, database_=settings.neo4j_database,
    ).records
    doc_ids = [r["doc_id"] for r in recs]
    periods = sorted({p for r in recs for p in r["periods"] if p})
    if not doc_ids:
        return None
    return {"candidate_doc_ids": doc_ids, "periods": periods}


def _api_correlations() -> list[dict]:
    recs = driver().execute_query(
        _API_CORR_CYPHER, database_=settings.neo4j_database,
    ).records
    return [{"sql_column": r["sql_column"], "api_column": r["api_column"]} for r in recs]


def build_plan(intent: "Intent") -> dict:
    """Deterministic graph planning. No LLM. Returns a JSON-serializable Plan dict."""
    resolved = _resolve_values(intent.terms)

    scope = {"fiscal_year": intent.fiscal_year, "quarter": intent.quarter}
    sql_legs = []
    sales_dims = [r for r in resolved if r["source"] == "sales_pg"]
    if sales_dims:
        sql_legs.append({
            "source": "sales_pg",
            "fact_table": _SALES_FACT,
            "join_targets": _join_targets(_SALES_FACT, [r["table_id"] for r in sales_dims]),
            "filters": [{"table_id": r["table_id"], "column": r["column"], "value": r["exact"]}
                        for r in sales_dims],
            "scope": scope,
        })
    if intent.financial_metrics:
        sql_legs.append({
            "source": "financials",
            "fact_table": "table:financials.main.income_statement",
            "join_targets": [], "filters": [], "scope": scope,
            "metrics": intent.financial_metrics,
        })

    doc_leg = None
    if intent.needs_doc:
        ctx = _context_docs(intent.terms)
        if ctx is not None:
            doc_leg = {"doc_query": intent.doc_query, **ctx}

    api_correlations = _api_correlations() if intent.needs_api else []

    highlight = sorted({
        *(r["table_id"] for r in resolved),
        *(t for leg in sql_legs for jt in leg["join_targets"] for t in jt["tables"]),
        *(doc_leg["candidate_doc_ids"] if doc_leg else []),
    })

    return {
        "resolved_values": resolved,
        "sql_legs": sql_legs,
        "doc_leg": doc_leg,
        "api_correlations": api_correlations,
        "highlight": highlight,
    }
