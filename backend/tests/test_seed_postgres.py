import psycopg
import pytest

from data.seed_postgres import seed

DEEP_JOIN_SQL = """
SELECT COUNT(*) AS line_count, COALESCE(SUM(ol.amount), 0) AS revenue
FROM sales.order_line ol
JOIN sales.product p           ON p.product_id = ol.product_id
JOIN sales.product_line pl     ON pl.product_line_id = p.product_line_id
JOIN sales.segment s           ON s.segment_id = pl.segment_id
JOIN sales.architecture a      ON a.architecture_id = pl.architecture_id
JOIN sales.sales_order so      ON so.order_id = ol.order_id
JOIN sales.fiscal_period fp    ON fp.fiscal_period_id = so.fiscal_period_id
JOIN sales.customer c          ON c.customer_id = so.customer_id
JOIN sales.industry i          ON i.industry_id = c.industry_id
JOIN sales.country co          ON co.country_id = c.country_id
JOIN sales.region r            ON r.region_id = co.region_id
WHERE s.name = 'Data Center' AND a.name = 'Blackwell';
"""


@pytest.mark.postgres
def test_seed_loads_all_tables(postgres_dsn):
    counts = seed(dsn=postgres_dsn)
    assert counts["sales.order_line"] >= 300
    with psycopg.connect(postgres_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM sales.product;")
        assert cur.fetchone()[0] == 20


@pytest.mark.postgres
def test_deep_eleven_table_join_returns_rows(postgres_dsn):
    seed(dsn=postgres_dsn)
    with psycopg.connect(postgres_dsn) as conn, conn.cursor() as cur:
        cur.execute(DEEP_JOIN_SQL)
        line_count, revenue = cur.fetchone()
    assert line_count > 0
    assert revenue > 0
