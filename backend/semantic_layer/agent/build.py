"""Assemble the deepagents orchestrator with sql/api/doc subagents."""

from deepagents import create_deep_agent

from semantic_layer.config import settings
from semantic_layer.ingest.llm import get_chat_model  # also ensures OPENAI_API_KEY in env
from semantic_layer.agent.graph_tools import (
    list_sources, list_tables, get_table_schema, get_join_path, search_catalog,
    resolve_value, neighbors, periods_for_documents,
)
from semantic_layer.agent.doc_tools import search_documents
from semantic_layer.agent.sql_tools import run_sql
from semantic_layer.agent.api_tools import call_api

_ORCHESTRATOR_PROMPT = """You answer questions over an NVIDIA enterprise semantic layer
that unifies SQL databases, REST APIs, and documents. Workflow:
1. Use search_catalog and list_sources to find which sources/tables are relevant.
2. For structured data, use get_table_schema and get_join_path to plan the query,
   then delegate to the 'sql' subagent with the exact tables, join path, and sql_reference.
3. For enterprise-system data (CRM, support tickets, partner inventory, DGX usage),
   delegate to the 'api' subagent.
4. For narrative/press-release questions, delegate to the 'doc' subagent.
5. A question may need several subagents; combine their results.
Always state which source(s) the answer came from. Be concise and cite documents by id.

ALIGNING TIME ACROSS DOCS AND SQL. Documents report a specific fiscal quarter while
order_line facts are all-time. When an answer combines a document with a SQL number,
call periods_for_documents([doc_id, ...]) on the documents you used; if it returns a
period with sql_available:true, tell the sql subagent to SCOPE its aggregation to that
fiscal_year and quarter so the two sides are comparable. If sql_available is false, do
NOT invent a filter — report the sql number as all-time and state the period mismatch.

ROUTING DATA VALUES. The graph indexes dimension row-values as :Value nodes, so
search_catalog now also returns hits with kind:"value" — each gives the exact stored
spelling plus the table_id and column that holds it (e.g. 'Blackwell' -> architecture,
'Data Center' -> segment, 'Cloud' -> industry='Cloud Service Provider', 'EMEA' ->
region). PREFER these value hits to map a filter to its table/column. neighbors(name)
shows a value's full cross-source neighborhood (which tables contain it AND which
documents mention it) — use it to link the document and structured worlds.

If a value hit is missing, fall back to the manual path. The catalog stores metadata,
not row values, so plain name search can miss the table. Values like 'EMEA', 'APAC',
'Cloud', 'Healthcare', 'Data Center', 'Gaming', 'Blackwell', or 'Hopper' are rows in
lookup/dimension tables (region, industry, segment, architecture) whose names never
appear in the question. For a question with such filters:
  a. call list_tables(source) on the relevant SQL source to see its FULL schema;
  b. DECOMPOSE the question into atomic descriptor tokens. A compound noun phrase is
     almost always SEVERAL independent filters on DIFFERENT dimensions, not one product
     name — split it. E.g. 'Blackwell Data Center products' = 'Blackwell' (architecture)
     AND 'Data Center' (segment), TWO separate filters; 'EMEA Cloud customers' = 'EMEA'
     (region) AND 'Cloud' (industry). Never merge adjacent descriptors into a single
     value or assume one implies the other;
  c. call resolve_value(value) on EACH atomic token — it returns which table/column holds
     that value and its EXACT stored spelling (e.g. 'Cloud' -> industry.name = 'Cloud
     Service Provider', 'Data Center' -> segment.name = 'Data Center'). Do NOT guess which
     dimension a value belongs to; resolve_value tells you authoritatively;
  d. APPLY EVERY resolved filter in the query — one resolved value per descriptor, ANDed
     together. Do not drop a filter just because another value seems related (e.g. keep
     BOTH segment='Data Center' AND architecture='Blackwell'). Use get_join_path to fold
     each dimension table into the join, and hand the sql subagent the full list of
     resolved table.column = 'exact value' filters.
Do not conclude data is missing until you have enumerated the source's tables AND
resolved every descriptor with resolve_value. Briefly list the filters you applied."""

_SQL_PROMPT = """You are a SQL expert. You are given the relevant tables, their
sql_reference values, and a join path. Write ONE read-only SELECT and run it with
run_sql(source, sql). For 'sales_pg', tables live under schema 'sales' (use the
sql_reference, e.g. sales.order_line). Use the provided join path to JOIN correctly,
including deep multi-table joins.
When filtering on a lookup/dimension NAME column (region, industry, segment,
architecture, etc.), the question usually gives a SHORTHAND of the stored value —
e.g. 'Cloud' for 'Cloud Service Provider', 'Ada' for 'Ada Lovelace', 'Prof Viz' for
'Professional Visualization'. So match those filters case-insensitively and partially
with ILIKE '%term%' rather than '=', unless the question gives the exact full value.
If a query returns zero rows, first run a quick SELECT DISTINCT on the suspect name
column to see the actual stored values, then adjust the filter and retry.
SCOPING TO A FISCAL PERIOD. If you are asked to scope results to a fiscal_year/quarter,
order_line has no date column — reach the time dimension by joining
order_line -> sales_order (order_id) -> fiscal_period (fiscal_period_id) and filter
WHERE fiscal_period.fiscal_year = <year> AND fiscal_period.quarter = '<Qn>'. Use
get_join_path(order_line, fiscal_period) to confirm the join. If the scoped query
returns zero rows, re-run WITHOUT the period filter and clearly report both the
all-time number and that no rows exist for the requested period.
If run_sql returns an error, fix the SQL and retry once. Report the rows and the SQL you ran."""

_API_PROMPT = """You call mock enterprise REST APIs with call_api(source, path, params).
Sources: crm (/accounts,/contacts,/opportunities), itsm (/tickets,/rma),
partner (/partners,/inventory), dgx (/usage). Pick the endpoint and query params that
answer the question, call it, and summarize the JSON. account_id == the sales customer id."""

_DOC_PROMPT = """You answer from NVIDIA documents using search_documents(query). Retrieve
passages, then answer ONLY from them, quoting the most relevant sentence and citing the
document id. If nothing relevant is found, say so."""


_ORCHESTRATOR_TOOLS = [
    list_sources, list_tables, resolve_value, neighbors, periods_for_documents,
    search_catalog, get_table_schema, get_join_path,
]


def build_agent():
    model = get_chat_model()  # gpt-5.4-mini; also sets OPENAI_API_KEY in env
    subagents = [
        {"name": "sql", "description": "Runs read-only SQL over the Postgres/SQLite sources.",
         "system_prompt": _SQL_PROMPT, "tools": [get_table_schema, get_join_path, run_sql],
         "model": settings.llm_model},
        {"name": "api", "description": "Calls the CRM/ITSM/partner/DGX mock REST APIs.",
         "system_prompt": _API_PROMPT, "tools": [get_table_schema, call_api],
         "model": settings.llm_model},
        {"name": "doc", "description": "Answers from the NVIDIA documents via vector search.",
         "system_prompt": _DOC_PROMPT, "tools": [search_documents], "model": settings.llm_model},
    ]
    return create_deep_agent(
        model=model,
        tools=_ORCHESTRATOR_TOOLS,
        system_prompt=_ORCHESTRATOR_PROMPT,
        subagents=subagents,
    )


def ask(question: str) -> str:
    agent = build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content
