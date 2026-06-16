import os

import psycopg
import pytest

from semantic_layer.config import settings


@pytest.fixture(scope="session", autouse=True)
def _propagate_openai_key():
    """Export the .env OPENAI_API_KEY into the process env so the OpenAI SDK,
    LangChain, and the require_openai gate all see it (pydantic Settings reads
    .env but does not populate os.environ)."""
    if settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    yield


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
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
