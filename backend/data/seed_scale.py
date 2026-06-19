"""Materialize the distractor catalog as EMPTY Postgres tables (catalog-only) and
re-seed the answerable core at scale volume. Distractor schemas are namespaced
`scale_*` and fully removable via drop_scale_schemas()."""

import psycopg

from data.generators.scale_catalog import ScaleCatalog, TableDef, generate_scale_catalog
from data.seed_postgres import seed as seed_core
from semantic_layer.config import settings


def _create_table_sql(t: TableDef) -> str:
    cols = []
    for c in t.columns:
        parts = [f'"{c.name}"', c.type]
        if c.is_pk:
            parts.append("PRIMARY KEY")
        if c.ref:
            ref_schema, ref_table, ref_col = c.ref.split(".")
            parts.append(f'REFERENCES {ref_schema}."{ref_table}" ("{ref_col}")')
        cols.append(" ".join(parts))
    body = ",\n  ".join(cols)
    return f'CREATE TABLE IF NOT EXISTS {t.schema}."{t.name}" (\n  {body}\n);'


def create_distractor_tables(dsn: str, catalog: ScaleCatalog) -> int:
    """Create every catalog table (empty) in FK-valid order. Idempotent."""
    schemas = sorted({t.schema for t in catalog.tables})
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        for s in schemas:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {s}")
        for t in catalog.tables:          # generator guarantees topological order
            cur.execute(_create_table_sql(t))
        conn.commit()
    return len(catalog.tables)


def drop_scale_schemas(dsn: str) -> None:
    """Drop all scale_* schemas, restoring the baseline DB."""
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT schema_name FROM information_schema.schemata WHERE schema_name LIKE 'scale_%'"
        )
        for (name,) in cur.fetchall():
            cur.execute(f"DROP SCHEMA {name} CASCADE")
        conn.commit()


def seed_scale(dsn: str | None = None, seed_value: int | None = None) -> dict:
    dsn = dsn or settings.postgres_dsn
    core = seed_core(
        dsn=dsn, seed_value=seed_value,
        n_customers=settings.scale_core_customers,
        n_orders=settings.scale_core_orders,
    )
    catalog = generate_scale_catalog(
        seed=settings.random_seed,
        n_tables=settings.scale_n_tables,
        n_apis=settings.scale_n_apis,
    )
    n = create_distractor_tables(dsn, catalog)
    return {"core_tables": len(core), "distractor_tables": n}


if __name__ == "__main__":
    print(seed_scale())
