import pytest

from semantic_layer.web.events import stream_chat_events

# A value-filtered question the graph-native planner resolves to a SQL leg.
_Q = "In FY2025, which EMEA Cloud customers bought Blackwell Data Center products?"


@pytest.mark.neo4j
@pytest.mark.postgres
@pytest.mark.openai
def test_chat_uses_controller_and_emits_final_answer(ingested_graph, require_openai):
    events = list(stream_chat_events(_Q))
    types = [e["type"] for e in events]
    assert types[-1] == "answer"
    # The controller surfaces a plan_query event (the graph-native plan) and leg results.
    assert any(e.get("name") == "plan_query" for e in events)
    assert "tool_result" in types
    answer = events[-1]
    assert isinstance(answer["highlight"], list)
    assert any(nid.startswith("table:sales_pg") for nid in answer["highlight"])


@pytest.mark.neo4j
@pytest.mark.postgres
@pytest.mark.openai
def test_answer_event_carries_sql_provenance(ingested_graph, require_openai):
    events = list(stream_chat_events(_Q))
    answer = events[-1]
    assert answer["type"] == "answer"
    # The SQL leg is captured with its query text and the structured provenance fields exist.
    assert isinstance(answer["sql_runs"], list) and answer["sql_runs"]
    run = answer["sql_runs"][0]
    assert run["sql"].strip().lower().startswith(("select", "with"))
    assert run["columns"]
    assert "caveats" in answer and isinstance(answer["caveats"], list)
    assert "doc_citations" in answer and "api_calls" in answer
