import json

from semantic_layer.web.events import _Provenance


def test_records_sql_run_with_sql_and_rows():
    prov = _Provenance()
    prov.record("run_sql", {"source": "sales_pg", "sql": "SELECT 1"},
                json.dumps({"columns": ["n"], "rows": [[1], [2]]}))
    fields = prov.answer_fields("ok")
    assert len(fields["sql_runs"]) == 1
    run = fields["sql_runs"][0]
    assert run["source"] == "sales_pg"
    assert run["sql"] == "SELECT 1"
    assert run["columns"] == ["n"]
    assert run["row_count"] == 2
    assert run["error"] is None


def test_records_sql_error():
    prov = _Provenance()
    prov.record("run_sql", {"source": "org", "sql": "SELECT bad"},
                json.dumps({"error": "no such column: bad"}))
    run = prov.answer_fields("ok")["sql_runs"][0]
    assert run["error"] == "no such column: bad"
    assert run["rows"] == []


def test_records_api_call():
    prov = _Provenance()
    prov.record("call_api", {"source": "itsm", "path": "/tickets", "params": {"status": "open"}},
                json.dumps({"status": 200, "data": [{"id": 1}, {"id": 2}]}))
    call = prov.answer_fields("ok")["api_calls"][0]
    assert call["source"] == "itsm"
    assert call["path"] == "/tickets"
    assert call["row_count"] == 2


def test_dedupes_doc_citations_by_chunk():
    prov = _Provenance()
    payload = json.dumps([
        {"chunk_id": "doc:x:chunk:1", "doc_id": "doc:x", "text": "hello", "score": 0.9},
        {"chunk_id": "doc:x:chunk:1", "doc_id": "doc:x", "text": "hello", "score": 0.9},
    ])
    prov.record("search_documents", {"query": "q"}, payload)
    cites = prov.answer_fields("ok")["doc_citations"]
    assert len(cites) == 1
    assert cites[0]["chunk_id"] == "doc:x:chunk:1"
    assert cites[0]["quote"] == "hello"


def test_answer_fields_runs_grounding():
    prov = _Provenance()
    prov.record("run_sql", {"source": "sales_pg", "sql": "SELECT 1"},
                json.dumps({"columns": ["n"], "rows": [[5]]}))
    fields = prov.answer_fields("The number is 99999.")
    assert any("99999" in c for c in fields["caveats"])
