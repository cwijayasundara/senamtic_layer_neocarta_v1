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


@pytest.fixture
def ingested_graph(neo4j_driver, postgres_dsn):
    """Ensure the metadata + document graph (no LLM) is present for agent tool tests.

    Function-scoped and rebuild-if-missing: other tests (e.g. doc-tool tests) reset
    the shared Neo4j graph, so we re-ingest only when the metadata layer is absent.
    When already present the check is instant."""
    from semantic_layer.config import settings
    from semantic_layer.ingest.pipeline import run_ingest
    with neo4j_driver.session(database=settings.neo4j_database) as session:
        tables = session.run("MATCH (t:Table) RETURN count(t) AS c").single()["c"]
        values = session.run("MATCH (v:Value) RETURN count(v) AS c").single()["c"]
    # Rebuild when the metadata layer OR the value layer is absent (the latter is
    # written by index_values, which runs even in the no-LLM ingest path).
    if tables < 11 or values == 0:
        run_ingest(with_llm=False, reset=True)
    return neo4j_driver
