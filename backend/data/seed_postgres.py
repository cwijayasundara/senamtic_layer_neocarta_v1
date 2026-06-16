"""Create the sales schema and load generated rows into Postgres."""

from pathlib import Path

import psycopg

from semantic_layer.config import settings
from data.generators.sales import generate_sales

_SCHEMA_FILE = Path(__file__).parent / "schema" / "postgres_sales.sql"

# (table name, generate_sales() output key, ordered column list).
# Insertion order matches FK dependencies (parents before children). Keeping the
# table, its data key, and its columns in one tuple makes it impossible to update
# one without the others.
_TABLES = [
    ("sales.region", "regions", ["region_id", "name"]),
    ("sales.country", "countries", ["country_id", "name", "iso_code", "region_id"]),
    ("sales.industry", "industries", ["industry_id", "name"]),
    ("sales.customer", "customers", ["customer_id", "name", "country_id", "industry_id"]),
    ("sales.segment", "segments", ["segment_id", "name"]),
    ("sales.architecture", "architectures", ["architecture_id", "name", "launch_year"]),
    ("sales.product_line", "product_lines", ["product_line_id", "name", "segment_id", "architecture_id"]),
    ("sales.product", "products", ["product_id", "product_line_id", "sku", "name", "msrp", "launch_date"]),
    ("sales.fiscal_period", "fiscal_periods", ["fiscal_period_id", "fiscal_year", "quarter", "start_date", "end_date"]),
    ("sales.sales_order", "sales_orders", ["order_id", "customer_id", "fiscal_period_id", "order_date"]),
    ("sales.order_line", "order_lines", ["line_id", "order_id", "product_id", "quantity", "unit_price", "amount"]),
]


def seed(dsn: str | None = None, seed_value: int | None = None) -> dict:
    dsn = dsn or settings.postgres_dsn
    data = generate_sales(seed=seed_value if seed_value is not None else settings.random_seed)
    counts = {}
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA_FILE.read_text())
            for table, data_key, cols in _TABLES:
                rows = data[data_key]
                placeholders = ", ".join(["%s"] * len(cols))
                stmt = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
                # executemany raises on any row error before we commit, so the
                # len(rows) count below reflects committed rows (no ON CONFLICT).
                cur.executemany(stmt, [[r[c] for c in cols] for r in rows])
                counts[table] = len(rows)
        conn.commit()
    return counts


if __name__ == "__main__":
    result = seed()
    for table, n in result.items():
        print(f"{table}: {n} rows")
