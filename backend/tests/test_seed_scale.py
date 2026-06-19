# backend/tests/test_seed_scale.py
import psycopg
import pytest

from data.generators.scale_catalog import generate_scale_catalog
from data.seed_scale import create_distractor_tables, drop_scale_schemas


@pytest.mark.postgres
def test_create_and_drop_distractor_tables(postgres_dsn):
    cat = generate_scale_catalog(seed=42, n_tables=30, n_apis=3)
    try:
        n = create_distractor_tables(postgres_dsn, cat)
        assert n == 30
        with psycopg.connect(postgres_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema LIKE 'scale_%'"
            )
            assert cur.fetchone()[0] == 30
            # tables are empty (catalog-only)
            t = cat.tables[0]
            cur.execute(f'SELECT count(*) FROM {t.schema}."{t.name}"')
            assert cur.fetchone()[0] == 0
    finally:
        drop_scale_schemas(postgres_dsn)
    with psycopg.connect(postgres_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM information_schema.schemata WHERE schema_name LIKE 'scale_%'"
        )
        assert cur.fetchone()[0] == 0
