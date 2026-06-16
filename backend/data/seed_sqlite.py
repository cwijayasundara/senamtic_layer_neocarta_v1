"""Build the financials.db and org.db SQLite source databases."""

import sqlite3
from pathlib import Path

from semantic_layer.config import settings
from data.generators.financials import generate_financials
from data.generators.org import generate_org

_FINANCIALS_DDL = """
DROP TABLE IF EXISTS income_statement;
CREATE TABLE income_statement (
    period_id        INTEGER PRIMARY KEY,
    fiscal_year      INTEGER NOT NULL,
    quarter          TEXT NOT NULL,
    revenue          REAL NOT NULL,
    gross_margin_pct REAL NOT NULL,
    operating_income REAL NOT NULL,
    net_income       REAL NOT NULL,
    eps              REAL NOT NULL
);
DROP TABLE IF EXISTS stock_price;
CREATE TABLE stock_price (
    price_date TEXT PRIMARY KEY,
    open       REAL NOT NULL,
    high       REAL NOT NULL,
    low        REAL NOT NULL,
    close      REAL NOT NULL,
    volume     INTEGER NOT NULL
);
"""

_ORG_DDL = """
-- Drop in child -> parent order so FK enforcement (PRAGMA foreign_keys=ON)
-- does not reject dropping a parent table that a child still references
-- when re-seeding an existing database.
DROP TABLE IF EXISTS headcount;
DROP TABLE IF EXISTS department;
DROP TABLE IF EXISTS location;
CREATE TABLE department (
    department_id INTEGER PRIMARY KEY,
    name          TEXT NOT NULL
);
CREATE TABLE location (
    location_id INTEGER PRIMARY KEY,
    city        TEXT NOT NULL,
    country     TEXT NOT NULL,
    region      TEXT NOT NULL
);
CREATE TABLE headcount (
    snapshot_id    INTEGER PRIMARY KEY,
    department_id  INTEGER NOT NULL REFERENCES department(department_id),
    location_id    INTEGER NOT NULL REFERENCES location(location_id),
    fiscal_year    INTEGER NOT NULL,
    quarter        TEXT NOT NULL,
    employee_count INTEGER NOT NULL
);
"""


def _insert(con, table, cols, rows):
    placeholders = ", ".join(["?"] * len(cols))
    con.executemany(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})",
        [[r[c] for c in cols] for r in rows],
    )


def seed_all(out_dir: str | None = None, seed_value: int | None = None) -> dict:
    out = Path(out_dir or settings.sqlite_dir)
    out.mkdir(parents=True, exist_ok=True)
    sv = seed_value if seed_value is not None else settings.random_seed

    fin = generate_financials(seed=sv)
    fin_path = out / "financials.db"
    con = sqlite3.connect(fin_path)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(_FINANCIALS_DDL)
    _insert(con, "income_statement",
            ["period_id", "fiscal_year", "quarter", "revenue", "gross_margin_pct",
             "operating_income", "net_income", "eps"], fin["income_statement"])
    _insert(con, "stock_price",
            ["price_date", "open", "high", "low", "close", "volume"], fin["stock_prices"])
    con.commit()
    con.close()

    org = generate_org(seed=sv)
    org_path = out / "org.db"
    con = sqlite3.connect(org_path)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(_ORG_DDL)
    _insert(con, "department", ["department_id", "name"], org["departments"])
    _insert(con, "location", ["location_id", "city", "country", "region"], org["locations"])
    _insert(con, "headcount",
            ["snapshot_id", "department_id", "location_id", "fiscal_year",
             "quarter", "employee_count"], org["headcount"])
    con.commit()
    con.close()

    return {"financials": str(fin_path), "org": str(org_path)}


if __name__ == "__main__":
    result = seed_all()
    for name, path in result.items():
        print(f"{name}: {path}")
