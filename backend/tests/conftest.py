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


@pytest.fixture(scope="session")
def neo4j_driver():
    """Skip neo4j-marked tests if the docker neo4j is not reachable."""
    from semantic_layer.graph.client import get_driver
    try:
        driver = get_driver()
        driver.verify_connectivity()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Neo4j not available: {exc}")
    yield driver
    driver.close()


@pytest.fixture(scope="session")
def require_openai():
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
