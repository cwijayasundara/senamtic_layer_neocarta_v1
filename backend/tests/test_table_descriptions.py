from semantic_layer.ingest.embeddings import _table_embed_text
from semantic_layer.ingest.table_descriptions import TABLE_DESCRIPTIONS


def test_embed_text_without_description_is_unchanged():
    assert _table_embed_text("order_line", ["line_id", "amount"]) == \
        "order_line — columns: line_id, amount"
    assert _table_embed_text("region", []) == "region"


def test_embed_text_folds_in_description():
    out = _table_embed_text("order_line", ["amount"], "sales revenue line items")
    assert "sales revenue line items" in out
    assert out == "order_line — sales revenue line items — columns: amount"


def test_descriptions_cover_the_load_bearing_tables():
    assert "table:sales_pg.sales.order_line" in TABLE_DESCRIPTIONS
    assert "table:financials.main.income_statement" in TABLE_DESCRIPTIONS
    # order_line description must mention revenue (the whole point)
    assert "revenue" in TABLE_DESCRIPTIONS["table:sales_pg.sales.order_line"].lower()
    # income_statement must be marked NOT regional/per-order to push it away
    assert "not" in TABLE_DESCRIPTIONS["table:financials.main.income_statement"].lower()
    assert all(isinstance(v, str) and v for v in TABLE_DESCRIPTIONS.values())
