from data.generators.scale_catalog import generate_scale_catalog, TableDef, ApiDef


def test_generates_requested_table_count():
    cat = generate_scale_catalog(seed=42, n_tables=200, n_apis=10)
    assert len(cat.tables) == 200
    assert len(cat.apis) == 10
    assert all(isinstance(t, TableDef) for t in cat.tables)
    assert all(isinstance(a, ApiDef) for a in cat.apis)


def test_is_deterministic():
    a = generate_scale_catalog(seed=7, n_tables=120, n_apis=8)
    b = generate_scale_catalog(seed=7, n_tables=120, n_apis=8)
    assert [(t.schema, t.name) for t in a.tables] == [(t.schema, t.name) for t in b.tables]


def test_fks_reference_earlier_tables_only():
    cat = generate_scale_catalog(seed=42, n_tables=300, n_apis=5)
    seen = set()
    for t in cat.tables:
        for c in t.columns:
            if c.ref:
                ref_schema, ref_table, _ = c.ref.split(".")
                assert (ref_schema, ref_table) in seen, f"{t.schema}.{t.name} refs unseen {c.ref}"
        seen.add((t.schema, t.name))


def test_includes_near_miss_tables():
    cat = generate_scale_catalog(seed=42, n_tables=300, n_apis=5)
    names = {f"{t.schema}.{t.name}" for t in cat.tables}
    # near-miss tables collide with core keywords (revenue/order/customer)
    assert any("revenue" in n or "order" in n or "customer" in n for n in names)


def test_schemas_are_namespaced():
    cat = generate_scale_catalog(seed=42, n_tables=100, n_apis=5)
    assert all(t.schema.startswith("scale_") for t in cat.tables)
