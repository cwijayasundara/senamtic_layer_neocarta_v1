# backend/tests/test_pipeline_scale.py
from semantic_layer.ingest import pipeline


def test_scale_bundles_empty_when_disabled(monkeypatch):
    monkeypatch.setattr(pipeline.settings, "scale_mode", False)
    assert pipeline._scale_bundles() == []


def test_scale_bundles_built_when_enabled(monkeypatch):
    monkeypatch.setattr(pipeline.settings, "scale_mode", True)
    monkeypatch.setattr(pipeline.settings, "scale_n_tables", 12)
    monkeypatch.setattr(pipeline.settings, "scale_n_apis", 3)
    # extract_postgres needs a DB; stub it to isolate the assembly logic.
    captured = {}

    def fake_extract_postgres(dsn, source="sales_pg", schema_name="sales"):
        from semantic_layer.ingest.sql_extractor import SchemaBundle
        captured.setdefault("schemas", []).append(schema_name)
        return SchemaBundle()

    monkeypatch.setattr(pipeline, "extract_postgres", fake_extract_postgres)
    bundles = pipeline._scale_bundles()
    # one bundle per distractor schema (scale_*) + one per synthetic API (3)
    assert len(captured["schemas"]) >= 1
    assert all(s.startswith("scale_") for s in captured["schemas"])
    assert len(bundles) == len(captured["schemas"]) + 3
