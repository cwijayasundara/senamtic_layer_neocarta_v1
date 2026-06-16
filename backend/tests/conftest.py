import psycopg
import pytest

from semantic_layer.config import settings


@pytest.fixture(scope="session")
def postgres_dsn():
    """Skip postgres-marked tests if the docker postgres is not reachable."""
    dsn = settings.postgres_dsn
    try:
        with psycopg.connect(dsn, connect_timeout=3):
            pass
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres not available at {dsn}: {exc}")
    return dsn
