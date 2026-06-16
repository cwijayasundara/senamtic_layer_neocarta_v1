"""Assemble the deepagents orchestrator with sql/api/doc subagents."""

from deepagents import create_deep_agent

from semantic_layer.config import settings
from semantic_layer.ingest.llm import get_chat_model  # also ensures OPENAI_API_KEY in env
from semantic_layer.agent.graph_tools import (
    list_sources, get_table_schema, get_join_path, search_catalog,
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
Always state which source(s) the answer came from. Be concise and cite documents by id."""

_SQL_PROMPT = """You are a SQL expert. You are given the relevant tables, their
sql_reference values, and a join path. Write ONE read-only SELECT and run it with
run_sql(source, sql). For 'sales_pg', tables live under schema 'sales' (use the
sql_reference, e.g. sales.order_line). Use the provided join path to JOIN correctly,
including deep multi-table joins. If run_sql returns an error, fix the SQL and retry
once. Report the rows and the SQL you ran."""

_API_PROMPT = """You call mock enterprise REST APIs with call_api(source, path, params).
Sources: crm (/accounts,/contacts,/opportunities), itsm (/tickets,/rma),
partner (/partners,/inventory), dgx (/usage). Pick the endpoint and query params that
answer the question, call it, and summarize the JSON. account_id == the sales customer id."""

_DOC_PROMPT = """You answer from NVIDIA documents using search_documents(query). Retrieve
passages, then answer ONLY from them, quoting the most relevant sentence and citing the
document id. If nothing relevant is found, say so."""


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
        tools=[list_sources, search_catalog, get_table_schema, get_join_path],
        system_prompt=_ORCHESTRATOR_PROMPT,
        subagents=subagents,
    )


def ask(question: str) -> str:
    agent = build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content
