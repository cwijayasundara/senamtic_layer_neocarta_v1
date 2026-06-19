"""Curated one-line descriptions for the answerable-core tables, folded into the
table embedding text so semantic routing maps business words to the right table.

Only these tables are described; everything else (incl. the scale_* distractors)
falls back to name + column names. The two load-bearing entries disambiguate the
"revenue" trap: sales revenue lives in order_line.amount, NOT income_statement."""

TABLE_DESCRIPTIONS: dict[str, str] = {
    "table:sales_pg.sales.order_line":
        "sales revenue line items; amount is the line revenue (quantity x unit_price); "
        "the source for revenue by region, industry, segment, product, or period",
    "table:sales_pg.sales.sales_order":
        "customer sales orders; one row per order, links order lines to a customer and fiscal period",
    "table:sales_pg.sales.customer":
        "customers (accounts) that place sales orders; linked to country and industry",
    "table:sales_pg.sales.product":
        "products sold; each belongs to a product line",
    "table:sales_pg.sales.product_line":
        "product lines grouping products into a business segment",
    "table:sales_pg.sales.segment":
        "business segments (e.g. Data Center, Gaming) products belong to",
    "table:sales_pg.sales.region":
        "geographic sales regions (e.g. EMEA, Americas) reached via customer country",
    "table:sales_pg.sales.country":
        "countries, each mapped to a sales region",
    "table:sales_pg.sales.industry":
        "customer industries (verticals)",
    "table:sales_pg.sales.fiscal_period":
        "fiscal quarters/years used to scope sales by period",
    "table:financials.main.income_statement":
        "company-level reported quarterly financial statements (total revenue, net income); "
        "NOT per-order, per-customer, or regional — do not use for revenue by region/segment",
    "table:financials.main.stock_price":
        "daily company stock prices (open, high, low, close, volume)",
}
