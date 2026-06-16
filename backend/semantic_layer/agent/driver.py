"""Process-wide cached Neo4j driver for agent tools."""

from functools import lru_cache

from neo4j import Driver

from semantic_layer.graph.client import get_driver


@lru_cache
def driver() -> Driver:
    return get_driver()
