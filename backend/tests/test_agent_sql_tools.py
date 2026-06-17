import json

import pytest

from semantic_layer.agent.sql_tools import run_sql


@pytest.mark.postgres
def test_run_sql_postgres_deep_join(postgres_dsn):
    sql = """
    SELECT s.name AS segment, SUM(ol.amount) AS revenue
    FROM sales.order_line ol
    JOIN sales.product p ON p.product_id = ol.product_id
    JOIN sales.product_line pl ON pl.product_line_id = p.product_line_id
    JOIN sales.segment s ON s.segment_id = pl.segment_id
    GROUP BY s.name ORDER BY revenue DESC
    """
    out = json.loads(run_sql.invoke({"source": "sales_pg", "sql": sql}))
    assert "columns" in out and "rows" in out
    assert any(r[0] == "Data Center" for r in out["rows"])


def test_run_sql_rejects_writes():
    out = json.loads(run_sql.invoke({"source": "sales_pg", "sql": "DELETE FROM sales.region"}))
    assert "error" in out


def test_run_sql_sqlite(tmp_path):
    from data.seed_sqlite import seed_all
    import semantic_layer.agent.sql_tools as st
    seed_all(out_dir=str(tmp_path))
    out = json.loads(st._run("financials", "SELECT COUNT(*) FROM income_statement", base_dir=str(tmp_path)))
    assert out["rows"][0][0] == 13
