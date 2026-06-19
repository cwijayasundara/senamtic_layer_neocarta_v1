from semantic_layer.config import Settings


def test_scale_defaults_are_inert():
    s = Settings()
    assert s.scale_mode is False
    assert s.fake_embeddings is False
    assert s.scale_n_tables == 1000
    assert s.scale_n_apis == 46
    assert s.scale_core_customers == 5000
    assert s.scale_core_orders == 50000


def test_scale_mode_reads_env(monkeypatch):
    monkeypatch.setenv("SCALE_MODE", "true")
    monkeypatch.setenv("SCALE_N_TABLES", "250")
    s = Settings()
    assert s.scale_mode is True
    assert s.scale_n_tables == 250


def test_scale_mode_forces_schema_routing(monkeypatch):
    monkeypatch.setenv("SCALE_MODE", "true")
    s = Settings()
    assert s.schema_routing_enabled is True


def test_routing_stays_off_without_scale_mode():
    s = Settings()
    assert s.scale_mode is False
    assert s.schema_routing_enabled is False
