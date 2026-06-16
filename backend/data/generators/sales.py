"""Deterministic synthetic sales facts built on the curated dimensions."""

import random
from datetime import date, timedelta

from faker import Faker

from .reference import build_dimensions

_VARIANTS = ["", " SXM"]


def generate_sales(seed=42, n_customers=40, n_orders=300):
    rng = random.Random(seed)
    fake = Faker()
    fake.seed_instance(seed)

    dims = build_dimensions()

    # --- products: two variants per product line ---
    products = []
    pid = 0
    dc_segment_id = next(s["segment_id"] for s in dims["segments"] if s["name"] == "Data Center")
    seg_by_line = {pl["product_line_id"]: pl["segment_id"] for pl in dims["product_lines"]}
    for pl in dims["product_lines"]:
        for variant in _VARIANTS:
            pid += 1
            base = 28000 if seg_by_line[pl["product_line_id"]] == dc_segment_id else 1500
            msrp = base + rng.randint(0, 12000)
            products.append(
                {
                    "product_id": pid,
                    "product_line_id": pl["product_line_id"],
                    "sku": f"NV-{pl['name'].replace(' ', '')}-{pid:03d}",
                    "name": (pl["name"] + variant).strip(),
                    "msrp": msrp,
                    "launch_date": f"{2020 + rng.randint(0, 4)}-{rng.randint(1, 12):02d}-15",
                }
            )

    # --- customers ---
    country_ids = [c["country_id"] for c in dims["countries"]]
    industry_ids = [i["industry_id"] for i in dims["industries"]]
    customers = []
    for cid in range(1, n_customers + 1):
        customers.append(
            {
                "customer_id": cid,
                "name": fake.unique.company(),
                "country_id": rng.choice(country_ids),
                "industry_id": rng.choice(industry_ids),
            }
        )

    # --- orders ---
    customer_ids = [c["customer_id"] for c in customers]
    fiscal_period_ids = [fp["fiscal_period_id"] for fp in dims["fiscal_periods"]]
    fp_by_id = {fp["fiscal_period_id"]: fp for fp in dims["fiscal_periods"]}
    sales_orders = []
    for oid in range(1, n_orders + 1):
        fpid = rng.choice(fiscal_period_ids)
        period = fp_by_id[fpid]
        start = date.fromisoformat(period["start_date"])
        end = date.fromisoformat(period["end_date"])
        order_day = start + timedelta(days=rng.randint(0, (end - start).days))
        sales_orders.append(
            {
                "order_id": oid,
                "customer_id": rng.choice(customer_ids),
                "fiscal_period_id": fpid,
                "order_date": order_day.isoformat(),
            }
        )

    # --- order lines: 1-3 per order ---
    product_index = [p["product_id"] for p in products]
    msrp_by_product = {p["product_id"]: p["msrp"] for p in products}
    order_lines = []
    lid = 0
    for o in sales_orders:
        for _ in range(rng.randint(1, 3)):
            lid += 1
            prod = rng.choice(product_index)
            qty = rng.randint(1, 64)
            unit_price = round(msrp_by_product[prod] * rng.uniform(0.85, 1.0), 2)
            order_lines.append(
                {
                    "line_id": lid,
                    "order_id": o["order_id"],
                    "product_id": prod,
                    "quantity": qty,
                    "unit_price": unit_price,
                    "amount": round(qty * unit_price, 2),
                }
            )

    return {**dims, "products": products, "customers": customers,
            "sales_orders": sales_orders, "order_lines": order_lines}
