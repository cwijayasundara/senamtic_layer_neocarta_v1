"""Create the sales schema and load generated rows into Postgres."""

from pathlib import Path

import psycopg

from semantic_layer.config import settings
from data.generators.sales import generate_sales

_SCHEMA_FILE = Path(__file__).parent / "schema" / "postgres_sales.sql"

# table name -> ordered column list (insertion order matches FK dependencies)
_TABLES = [
    ("sales.region", ["region_id", "name"]),
    ("sales.country", ["country_id", "name", "iso_code", "region_id"]),
    ("sales.industry", ["industry_id", "name"]),
    ("sales.customer", ["customer_id", "name", "country_id", "industry_id"]),
    ("sales.segment", ["segment_id", "name"]),
    ("sales.architecture", ["architecture_id", "name", "launch_year"]),
    ("sales.product_line", ["product_line_id", "name", "segment_id", "architecture_id"]),
    ("sales.product", ["product_id", "product_line_id", "sku", "name", "msrp", "launch_date"]),
    ("sales.fiscal_period", ["fiscal_period_id", "fiscal_year", "quarter", "start_date", "end_date"]),
    ("sales.sales_order", ["order_id", "customer_id", "fiscal_period_id", "order_date"]),
    ("sales.order_line", ["line_id", "order_id", "product_id", "quantity", "unit_price", "amount"]),
]

# data key per table (the generate_sales() output uses these keys)
_DATA_KEY = {
    "sales.region": "regions",
    "sales.country": "countries",
    "sales.industry": "industries",
    "sales.customer": "customers",
    "sales.segment": "segments",
    "sales.architecture": "architectures",
    "sales.product_line": "product_lines",
    "sales.product": "products",
    "sales.fiscal_period": "fiscal_periods",
    "sales.sales_order": "sales_orders",
    "sales.order_line": "order_lines",
}


def seed(dsn: str | None = None, seed_value: int | None = None) -> dict:
    dsn = dsn or settings.postgres_dsn
    data = generate_sales(seed=seed_value if seed_value is not None else settings.random_seed)
    counts = {}
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA_FILE.read_text())
            for table, cols in _TABLES:
                rows = data[_DATA_KEY[table]]
                placeholders = ", ".join(["%s"] * len(cols))
                stmt = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
                cur.executemany(stmt, [[r[c] for c in cols] for r in rows])
                counts[table] = len(rows)
        conn.commit()
    return counts


if __name__ == "__main__":
    result = seed()
    for table, n in result.items():
        print(f"{table}: {n} rows")
