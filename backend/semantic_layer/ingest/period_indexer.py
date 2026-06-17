"""Index the sales fiscal-period dimension into the graph as :Period nodes.

A document quotes a fiscal quarter ("Q1 Fiscal 2027") while order_line facts reach
time only via order_id -> sales_order.fiscal_period_id -> fiscal_period. This stage
turns each fiscal_period row into a canonical (:Period {key}) node carrying its
fiscal_period_id, so a document's COVERS_PERIOD edge (doc_graph.link_document_period)
converges on the SQL-backed node and a SQL aggregation can be scoped to the same
window. Pure read-only SQL — runs on every ingest. fiscal_period lives only in
sales_pg, so the source is fixed.
"""

import json

from neo4j import Driver

from semantic_layer.agent.sql_tools import _run
from semantic_layer.config import settings
from semantic_layer.ingest.doc_graph import period_key

_SOURCE = "sales_pg"
_SQL = ("SELECT fiscal_period_id, fiscal_year, quarter, start_date, end_date "
        "FROM sales.fiscal_period")


def index_periods(driver: Driver) -> int:
    """MERGE a :Period node per sales.fiscal_period row. Returns the count written."""
    res = json.loads(_run(_SOURCE, _SQL))
    rows = res.get("rows")
    if not rows:
        return 0
    periods = [
        {
            "key": period_key(r[1], r[2]),
            "fiscal_period_id": r[0],
            "fiscal_year": r[1],
            "quarter": r[2],
            "start_date": str(r[3]),
            "end_date": str(r[4]),
        }
        for r in rows
    ]
    with driver.session(database=settings.neo4j_database) as session:
        session.run(
            """
            UNWIND $periods AS p
            MERGE (per:Period {key: p.key})
              SET per.fiscal_period_id = p.fiscal_period_id,
                  per.fiscal_year      = p.fiscal_year,
                  per.quarter          = p.quarter,
                  per.start_date       = p.start_date,
                  per.end_date         = p.end_date,
                  per.sql_reference    = 'sales.fiscal_period'
            """,
            periods=periods,
        )
    return len(periods)
