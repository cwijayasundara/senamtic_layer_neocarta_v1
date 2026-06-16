"""Neo4j driver factory and small graph utilities."""

from neo4j import Driver, GraphDatabase

from semantic_layer.config import settings


def get_driver() -> Driver:
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


def ping(driver: Driver) -> bool:
    driver.verify_connectivity()
    with driver.session(database=settings.neo4j_database) as session:
        return session.run("RETURN 1 AS ok").single()["ok"] == 1


def reset_graph(driver: Driver) -> None:
    """Delete all nodes and relationships. Used before a full re-ingest."""
    with driver.session(database=settings.neo4j_database) as session:
        session.run("MATCH (n) DETACH DELETE n")
