import pytest


@pytest.fixture
def agent_graph(ingested_graph):
    # ensure chunk embeddings exist so the doc subagent works
    from semantic_layer.ingest.embeddings import embed_chunks
    embed_chunks(ingested_graph)
    return ingested_graph


@pytest.mark.neo4j
@pytest.mark.postgres
@pytest.mark.openai
def test_structured_deep_join_question(agent_graph, require_openai):
    from semantic_layer.agent.build import ask
    answer = ask("Which business segment has the highest total revenue? Use the sales database.")
    assert "Data Center" in answer


@pytest.mark.neo4j
@pytest.mark.postgres
@pytest.mark.openai
def test_api_question(agent_graph, require_openai):
    from semantic_layer.agent.build import ask
    answer = ask("How many open support tickets are there? Use the support system.")
    assert any(ch.isdigit() for ch in answer)


@pytest.mark.neo4j
@pytest.mark.postgres
@pytest.mark.openai
def test_document_question(agent_graph, require_openai):
    from semantic_layer.agent.build import ask
    answer = ask("According to the press releases, what drove Data Center growth?")
    assert len(answer) > 0
