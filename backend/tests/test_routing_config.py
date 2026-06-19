from semantic_layer.config import Settings


def test_k_vec_default():
    assert Settings().schema_routing_k_vec == 30


def test_k_vec_reads_env(monkeypatch):
    monkeypatch.setenv("SCHEMA_ROUTING_K_VEC", "12")
    assert Settings().schema_routing_k_vec == 12
