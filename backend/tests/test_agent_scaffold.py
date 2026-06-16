import pytest

from semantic_layer.agent.driver import driver


@pytest.mark.neo4j
def test_driver_is_cached(neo4j_driver):
    assert driver() is driver()
