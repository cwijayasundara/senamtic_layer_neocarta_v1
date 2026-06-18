"""Read-only SQL execution tool with per-source engine routing."""

import json
import re
import sqlite3
from pathlib import Path

import psycopg
from langchain_core.tools import tool

from semantic_layer.agent.pg_pool import get_pool
from semantic_layer.config import settings

_READONLY = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_SQLITE_SOURCES = {"financials", "org"}


def _run(source: str, sql: str, base_dir: str | None = None,
         params: tuple | None = None) -> str:
    if not _READONLY.match(sql or ""):
        return json.dumps({"error": "only read-only SELECT/WITH queries are allowed"})
    limit = settings.agent_max_rows
    try:
        if source == "sales_pg":
            pool = get_pool()
            pool.open()  # idempotent; opens the pool lazily on first real use
            with pool.connection() as conn, conn.cursor() as cur:
                cur.execute(sql) if params is None else cur.execute(sql, params)
                cols = [d.name for d in cur.description]
                rows = cur.fetchmany(limit)
        elif source in _SQLITE_SOURCES:
            path = Path(base_dir or settings.sqlite_dir) / f"{source}.db"
            con = sqlite3.connect(path)
            try:
                cur = con.execute(sql) if params is None else con.execute(sql, params)
                cols = [d[0] for d in cur.description]
                rows = cur.fetchmany(limit)
            finally:
                con.close()
        else:
            return json.dumps({"error": f"unknown sql source '{source}'"})
    except Exception as exc:  # noqa: BLE001 — surface SQL errors back to the agent for self-repair
        return json.dumps({"error": str(exc)})
    return json.dumps({"columns": cols, "rows": [list(r) for r in rows]}, default=str)


@tool
def run_sql(source: str, sql: str) -> str:
    """Run a read-only SQL query against a structured source and return rows as JSON.

    source is one of 'sales_pg' (Postgres, tables under schema 'sales'),
    'financials', or 'org' (SQLite, unqualified table names). Only SELECT/WITH is
    allowed. On a SQL error the error text is returned so you can correct the query."""
    return _run(source, sql)
