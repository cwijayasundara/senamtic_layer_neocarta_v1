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


@pytest.mark.neo4j
@pytest.mark.postgres
@pytest.mark.openai
def test_answer_event_carries_sql_provenance(ingested_graph, require_openai):
    events = list(stream_chat_events(
        "Which segment has the most revenue? Use the sales database."))
    answer = events[-1]
    assert answer["type"] == "answer"
    # New structured fields exist and the SQL run was captured with its query text.
    assert isinstance(answer["sql_runs"], list) and answer["sql_runs"]
    run = answer["sql_runs"][0]
    assert run["sql"].strip().lower().startswith(("select", "with"))
    assert run["columns"]
    assert run["row_count"] >= 1
    assert "caveats" in answer and isinstance(answer["caveats"], list)
    assert "doc_citations" in answer and "api_calls" in answer
