import pytest

from semantic_layer.web.events import stream_chat_events


@pytest.mark.neo4j
@pytest.mark.postgres
@pytest.mark.openai
def test_stream_emits_tools_and_final_answer(ingested_graph, require_openai):
    events = list(stream_chat_events("Which segment has the most revenue? Use the sales database."))
    types = [e["type"] for e in events]
    assert "tool_call" in types
    assert types[-1] == "answer"
    answer = events[-1]
    assert "Data Center" in answer["content"]
    assert isinstance(answer["highlight"], list)
    assert any(nid.startswith("table:sales_pg") for nid in answer["highlight"])
    assert any(e["type"] == "tool_result" and e["name"] == "run_sql" for e in events)
