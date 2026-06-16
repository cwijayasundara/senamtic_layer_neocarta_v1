from data.generators.sales import generate_sales


def test_generation_is_deterministic():
    a = generate_sales(seed=42)
    b = generate_sales(seed=42)
    assert a["order_lines"] == b["order_lines"]


def test_referential_integrity_products_to_product_lines():
    data = generate_sales(seed=42)
    pl_ids = {pl["product_line_id"] for pl in data["product_lines"]}
    assert all(p["product_line_id"] in pl_ids for p in data["products"])


def test_order_lines_reference_valid_orders_and_products():
    data = generate_sales(seed=42)
    order_ids = {o["order_id"] for o in data["sales_orders"]}
    product_ids = {p["product_id"] for p in data["products"]}
    for ol in data["order_lines"]:
        assert ol["order_id"] in order_ids
        assert ol["product_id"] in product_ids


def test_amount_equals_quantity_times_unit_price():
    data = generate_sales(seed=42)
    for ol in data["order_lines"]:
        assert ol["amount"] == round(ol["quantity"] * ol["unit_price"], 2)


def test_volumes_are_reasonable():
    data = generate_sales(seed=42, n_customers=40, n_orders=300)
    assert len(data["products"]) == 20          # 2 variants per product line
    assert len(data["customers"]) == 40
    assert len(data["sales_orders"]) == 300
    assert len(data["order_lines"]) >= 300
