import pytest

from semantic_layer.agent.driver import driver


@pytest.mark.neo4j
def test_driver_is_cached(neo4j_driver):
    assert driver() is driver()


def test_orchestrator_exposes_periods_for_documents():
    # The orchestrator needs the period tool to scope SQL to a document's quarter.
    from semantic_layer.agent.build import _ORCHESTRATOR_TOOLS
    assert "periods_for_documents" in {t.name for t in _ORCHESTRATOR_TOOLS}
