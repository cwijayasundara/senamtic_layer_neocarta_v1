import pytest

from semantic_layer.ingest.sql_extractor import extract_postgres, extract_sqlite


@pytest.mark.postgres
def test_extract_postgres_sales_schema(postgres_dsn):
    bundle = extract_postgres(postgres_dsn, source="sales_pg")
    table_names = {t.name for t in bundle.tables}
    assert {"order_line", "product", "region"} <= table_names
    assert len(bundle.tables) == 11
    assert len(bundle.references) >= 10
    assert any(c.is_foreign_key for c in bundle.columns)
    assert any(c.is_primary_key for c in bundle.columns)


def test_extract_sqlite_financials(tmp_path):
    from data.seed_sqlite import seed_all
    seed_all(out_dir=str(tmp_path))
    bundle = extract_sqlite(str(tmp_path / "financials.db"), source="financials")
    names = {t.name for t in bundle.tables}
    assert {"income_statement", "stock_price"} <= names
    assert all(c.type for c in bundle.columns)
