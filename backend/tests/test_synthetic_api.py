from data.generators.scale_catalog import generate_scale_catalog
from semantic_layer.ingest.api_extractor import extract_openapi
from semantic_layer.ingest.synthetic_api import build_openapi_spec, extract_synthetic_apis


def test_spec_parses_into_virtual_table():
    cat = generate_scale_catalog(seed=42, n_tables=10, n_apis=2)
    api = cat.apis[0]
    spec = build_openapi_spec(api)
    bundle = extract_openapi(spec, source=api.source)
    table_names = {t.name for t in bundle.tables}
    assert f"GET /{api.resource}" in table_names
    col_names = {c.name for c in bundle.columns}
    assert {f.name for f in api.fields} <= col_names


def test_extract_synthetic_apis_one_bundle_per_api():
    cat = generate_scale_catalog(seed=42, n_tables=10, n_apis=5)
    bundles = extract_synthetic_apis(cat)
    assert len(bundles) == 5
    assert all(b.tables for b in bundles)
