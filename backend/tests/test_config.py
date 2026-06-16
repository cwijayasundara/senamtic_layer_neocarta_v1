from semantic_layer.config import Settings


def test_defaults_load_without_env_file():
    s = Settings(_env_file=None)
    assert s.postgres_db == "nvidia"
    assert s.postgres_port == 5432
    assert s.random_seed == 42
    assert s.sqlite_dir == "data/seed"


def test_postgres_dsn_is_well_formed():
    s = Settings(_env_file=None)
    assert s.postgres_dsn == (
        "postgresql://neocarta:neocarta@localhost:5432/nvidia"
    )
