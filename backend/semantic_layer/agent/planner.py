"""Graph-native planner: one LLM intent pass, then deterministic graph planning.

extract_intent(question) -> Intent     (one structured LLM call, planner_model)
build_plan(intent)       -> Plan dict   (pure Cypher; added in a later task)

This replaces the orchestrator's ~20 LLM discovery round-trips with a single intent
read plus a few set-based graph queries.
"""

import json
import re

from pydantic import BaseModel, Field

from semantic_layer.agent.driver import driver
from semantic_layer.agent.graph_tools import get_join_path
from semantic_layer.agent.routing import route_tables, select_fact_table
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


def _table_columns(table_id: str) -> list[str]:
    """Actual column names of a table, so the SQL leg uses real names (not guesses)."""
    recs = driver().execute_query(
        "MATCH (t:Table {id:$id})-[:HAS_COLUMN]->(c:Column) RETURN c.name AS name ORDER BY name",
        id=table_id, database_=settings.neo4j_database,
    ).records
    return [r["name"] for r in recs]


_TIME_TOKENS = {"quarter", "fiscal_year", "fiscal_period", "period", "year", "time"}


def _dimension_targets(group_by: list[str]) -> list[str]:
    """Map group-by tokens (e.g. 'segment','customer','quarter') to sales dimension tables."""
    out = []
    for g in group_by:
        key = norm(g)
        if key in _TIME_TOKENS:
            out.append("table:sales_pg.sales.fiscal_period")
            continue
        # Lenient: 'business segment' should match the 'segment' table; prefer the
        # longest table-name match so 'product line' beats 'product'.
        recs = driver().execute_query(
            "MATCH (t:Table) WHERE t.id STARTS WITH 'table:sales_pg.sales.' "
            "AND (toLower(t.name) = $n OR $n CONTAINS toLower(t.name)) "
            "RETURN t.id AS id ORDER BY size(t.name) DESC LIMIT 1",
            n=key, database_=settings.neo4j_database,
        ).records
        if recs:
            out.append(recs[0]["id"])
    return list(dict.fromkeys(out))


def build_plan(intent: "Intent", question: str | None = None) -> dict:
    """Deterministic graph planning. No LLM unless schema routing is enabled.
    Returns a JSON-serializable Plan dict."""
    resolved = _resolve_values(intent.terms)

    # Optional retrieve-then-rank routing: only sales-schema tables are foldable
    # into the deterministic sales leg here; other routed ids are still surfaced.
    routed_tables: list[str] = []
    if settings.schema_routing_enabled and question:
        routed_tables = route_tables(
            question, k_ret=settings.schema_routing_k_ret,
            k_rank=settings.schema_routing_k_rank)

    fact_table = _SALES_FACT
    if routed_tables:
        fact_table = select_fact_table(routed_tables) or _SALES_FACT

    # Document context first, so a question with no explicit period can scope SQL to the
    # period the cited press release reports ("compare with the latest release").
    doc_leg = None
    if intent.needs_doc:
        ctx = _context_docs(intent.terms)
        if ctx is not None:
            doc_leg = {"doc_query": intent.doc_query, **ctx}

    fiscal_year, quarter = intent.fiscal_year, intent.quarter
    if fiscal_year is None and doc_leg and doc_leg.get("periods"):
        m = re.match(r"FY(\d{4})-(Q\d)", sorted(doc_leg["periods"])[-1])
        if m:
            fiscal_year, quarter = int(m.group(1)), m.group(2)
    scope = {"fiscal_year": fiscal_year, "quarter": quarter}

    sql_legs = []
    sales_dims = [r for r in resolved if r["source"] == "sales_pg"]
    dim_targets = _dimension_targets(intent.group_by) if intent.needs_sql else []
    routed_sales = [t for t in routed_tables if t.startswith("table:sales_pg.sales.")]
    sales_target_ids = [r["table_id"] for r in sales_dims] + dim_targets + routed_sales
    # When routing is on, BOUND the join to the routed top-K so a large catalog yields a
    # bounded join rather than a whole-schema scan; preserve order, de-dupe, then cap.
    if settings.schema_routing_enabled and routed_sales:
        routed_set = set(routed_sales)
        bounded = [t for t in dict.fromkeys(sales_target_ids) if t in routed_set]
        sales_target_ids = bounded[: settings.schema_routing_max_targets]
    if sales_target_ids:
        sql_legs.append({
            "source": "sales_pg",
            "fact_table": fact_table,
            "join_targets": _join_targets(fact_table, sales_target_ids),
            "filters": [{"table_id": r["table_id"], "column": r["column"], "value": r["exact"]}
                        for r in sales_dims],
            "group_by": list(intent.group_by) if dim_targets else [],
            "scope": scope,
        })
    if intent.financial_metrics:
        sql_legs.append({
            "source": "financials",
            "fact_table": "table:financials.main.income_statement",
            "join_targets": [], "filters": [], "scope": scope,
            "metrics": intent.financial_metrics,
        })
    for leg in sql_legs:  # real column names so the leg writes valid SQL, not guesses
        leg["columns"] = _table_columns(leg["fact_table"])

    api_correlations = _api_correlations() if intent.needs_api else []

    highlight = sorted({
        *(r["table_id"] for r in resolved),
        *(t for leg in sql_legs for jt in leg["join_targets"] for t in jt["tables"]),
        *(doc_leg["candidate_doc_ids"] if doc_leg else []),
        *routed_tables,
    })

    return {
        "resolved_values": resolved,
        "sql_legs": sql_legs,
        "doc_leg": doc_leg,
        "api_correlations": api_correlations,
        "highlight": highlight,
        "routed_tables": routed_tables,
    }
