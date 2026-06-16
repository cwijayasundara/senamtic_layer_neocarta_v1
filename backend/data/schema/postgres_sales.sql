DROP SCHEMA IF EXISTS sales CASCADE;
CREATE SCHEMA sales;

CREATE TABLE sales.region (
    region_id   INTEGER PRIMARY KEY,
    name        TEXT NOT NULL
);

CREATE TABLE sales.country (
    country_id  INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    iso_code    TEXT NOT NULL,
    region_id   INTEGER NOT NULL REFERENCES sales.region(region_id)
);

CREATE TABLE sales.industry (
    industry_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL
);

CREATE TABLE sales.customer (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    country_id  INTEGER NOT NULL REFERENCES sales.country(country_id),
    industry_id INTEGER NOT NULL REFERENCES sales.industry(industry_id)
);

CREATE TABLE sales.segment (
    segment_id  INTEGER PRIMARY KEY,
    name        TEXT NOT NULL
);

CREATE TABLE sales.architecture (
    architecture_id INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    launch_year     INTEGER NOT NULL
);

CREATE TABLE sales.product_line (
    product_line_id INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    segment_id      INTEGER NOT NULL REFERENCES sales.segment(segment_id),
    architecture_id INTEGER NOT NULL REFERENCES sales.architecture(architecture_id)
);

CREATE TABLE sales.product (
    product_id      INTEGER PRIMARY KEY,
    product_line_id INTEGER NOT NULL REFERENCES sales.product_line(product_line_id),
    sku             TEXT NOT NULL,
    name            TEXT NOT NULL,
    msrp            NUMERIC(12,2) NOT NULL,
    launch_date     DATE NOT NULL
);

CREATE TABLE sales.fiscal_period (
    fiscal_period_id INTEGER PRIMARY KEY,
    fiscal_year      INTEGER NOT NULL,
    quarter          TEXT NOT NULL,
    start_date       DATE NOT NULL,
    end_date         DATE NOT NULL
);

CREATE TABLE sales.sales_order (
    order_id         INTEGER PRIMARY KEY,
    customer_id      INTEGER NOT NULL REFERENCES sales.customer(customer_id),
    fiscal_period_id INTEGER NOT NULL REFERENCES sales.fiscal_period(fiscal_period_id),
    order_date       DATE NOT NULL
);

CREATE TABLE sales.order_line (
    line_id    INTEGER PRIMARY KEY,
    order_id   INTEGER NOT NULL REFERENCES sales.sales_order(order_id),
    product_id INTEGER NOT NULL REFERENCES sales.product(product_id),
    quantity   INTEGER NOT NULL,
    unit_price NUMERIC(12,2) NOT NULL,
    amount     NUMERIC(14,2) NOT NULL
);
