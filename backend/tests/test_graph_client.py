import pytest

from semantic_layer.graph.client import get_driver, ping


@pytest.mark.neo4j
def test_driver_connects_and_pings(neo4j_driver):
    assert ping(neo4j_driver) is True


@pytest.mark.neo4j
def test_reset_graph_clears_nodes(neo4j_driver):
    from semantic_layer.graph.client import reset_graph
    with neo4j_driver.session() as s:
        s.run("CREATE (:Probe {k: 1})")
    reset_graph(neo4j_driver)
    with neo4j_driver.session() as s:
        count = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    assert count == 0
