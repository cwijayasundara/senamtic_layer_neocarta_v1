"""Deterministic generator for a large *distractor* catalog: synthetic business
domains of empty tables + synthetic REST APIs that surround the answerable core,
so schema routing must discriminate the real tables from realistic noise."""

import random
from dataclasses import dataclass, field


@dataclass
class ColumnDef:
    name: str
    type: str               # SQL type: INTEGER | TEXT | NUMERIC | DATE | BOOLEAN
    is_pk: bool = False
    ref: str | None = None  # FK target "schema.table.column", or None


@dataclass
class TableDef:
    schema: str             # e.g. "scale_hr"
    name: str               # e.g. "employee"
    columns: list[ColumnDef]


@dataclass
class ApiField:
    name: str
    type: str               # OpenAPI type: integer | string | number


@dataclass
class ApiDef:
    source: str             # e.g. "scale_hr_api"
    resource: str           # e.g. "employees"
    fields: list[ApiField]


@dataclass
class ScaleCatalog:
    tables: list[TableDef] = field(default_factory=list)
    apis: list[ApiDef] = field(default_factory=list)


# Realistic business domains. Each becomes a scale_<domain> schema. The names and
# noun pools are deliberately business-plausible so routing faces real distractors.
_DOMAINS = {
    "hr": ["employee", "department", "payroll", "benefit", "leave_request", "review"],
    "logistics": ["shipment", "warehouse", "route", "carrier", "manifest", "delivery"],
    "marketing": ["campaign", "lead", "channel", "spend", "impression", "conversion"],
    "iot": ["device", "sensor", "reading", "firmware", "alert", "gateway"],
    "supply_chain": ["supplier", "purchase_order", "material", "lot", "receipt", "forecast"],
    "finance_ops": ["invoice", "ledger_entry", "cost_center", "budget", "accrual", "tax_code"],
    "support_ops": ["case", "agent", "queue", "survey", "escalation", "knowledge_article"],
}

# Near-miss nouns collide with answerable-core terms (sales revenue/orders/customers).
_NEAR_MISS = {
    "marketing": [("campaign_revenue", "revenue")],
    "finance_ops": [("quarterly_revenue", "revenue"), ("customer_invoice", "customer")],
    "logistics": [("order_shipment", "order")],
}

_COL_POOL = [
    ("name", "TEXT"), ("status", "TEXT"), ("created_at", "DATE"),
    ("amount", "NUMERIC"), ("quantity", "INTEGER"), ("active", "BOOLEAN"),
    ("region", "TEXT"), ("notes", "TEXT"), ("code", "TEXT"),
]


def _columns_for(rng, schema, table, earlier):
    """Build columns: a PK, 2-4 attributes, and 0-1 FK to an earlier table."""
    cols = [ColumnDef(name=f"{table}_id", type="INTEGER", is_pk=True)]
    for cname, ctype in rng.sample(_COL_POOL, rng.randint(2, 4)):
        cols.append(ColumnDef(name=cname, type=ctype))
    if earlier and rng.random() < 0.6:
        tgt_schema, tgt_table = rng.choice(earlier)
        cols.append(ColumnDef(
            name=f"{tgt_table}_id", type="INTEGER",
            ref=f"{tgt_schema}.{tgt_table}.{tgt_table}_id",
        ))
    return cols


def generate_scale_catalog(seed: int = 42, n_tables: int = 1000, n_apis: int = 46) -> ScaleCatalog:
    rng = random.Random(seed)
    cat = ScaleCatalog()
    earlier: list[tuple[str, str]] = []          # (schema, table) already emitted
    domains = list(_DOMAINS.items())

    # Reserve slots for the fixed set of near-miss tables so they survive the final
    # count (they are appended after the round-robin fill).
    near_miss_total = sum(len(p) for p in _NEAR_MISS.values())
    target_round_robin = max(0, n_tables - near_miss_total)

    # Round-robin tables across domains, suffixing to reach the fill target uniquely.
    i = 0
    while len(cat.tables) < target_round_robin:
        domain, nouns = domains[i % len(domains)]
        schema = f"scale_{domain}"
        noun = nouns[(i // len(domains)) % len(nouns)]
        suffix = i // (len(domains) * len(nouns))
        name = noun if suffix == 0 else f"{noun}_{suffix}"
        cols = _columns_for(rng, schema, name, earlier)
        cat.tables.append(TableDef(schema=schema, name=name, columns=cols))
        earlier.append((schema, name))
        i += 1

    # Append near-miss tables so a few distractor names collide with core keywords.
    # Each references an earlier same-schema table when one exists.
    for domain, pairs in _NEAR_MISS.items():
        schema = f"scale_{domain}"
        prior = [(s, t) for (s, t) in earlier if s == schema]
        for nm_name, _kw in pairs:
            cols = [ColumnDef(name=f"{nm_name}_id", type="INTEGER", is_pk=True),
                    ColumnDef(name="amount", type="NUMERIC"),
                    ColumnDef(name="period", type="TEXT")]
            if prior:
                s, t = prior[0]
                cols.append(ColumnDef(name=f"{t}_id", type="INTEGER", ref=f"{s}.{t}.{t}_id"))
            cat.tables.append(TableDef(schema=schema, name=nm_name, columns=cols))
            earlier.append((schema, nm_name))

    # Safety: never exceed the requested count (near_miss_total < n_tables in practice).
    cat.tables = cat.tables[:n_tables]

    # Synthetic REST APIs: one resource each, drawn from the same domains.
    for j in range(n_apis):
        domain, nouns = domains[j % len(domains)]
        noun = nouns[j % len(nouns)]
        resource = f"{noun}s_{j}"
        fields = [ApiField(name=f"{noun}_id", type="integer"),
                  ApiField(name="name", type="string"),
                  ApiField(name="status", type="string"),
                  ApiField(name="amount", type="number")]
        cat.apis.append(ApiDef(source=f"scale_{domain}_api_{j}", resource=resource, fields=fields))

    return cat
