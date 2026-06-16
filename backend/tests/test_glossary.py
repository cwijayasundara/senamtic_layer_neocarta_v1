import pytest

from semantic_layer.ingest.glossary import generate_business_terms


@pytest.mark.openai
def test_generate_business_terms_for_columns(require_openai):
    columns = [
        {"column_id": "col:sales_pg.sales.order_line.amount", "name": "amount", "table": "order_line"},
        {"column_id": "col:sales_pg.sales.segment.name", "name": "name", "table": "segment"},
    ]
    terms = generate_business_terms(columns)
    assert len(terms) > 0
    for t in terms:
        assert t["name"] and t["description"]
        assert t["column_id"] in {c["column_id"] for c in columns}
