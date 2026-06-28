import json

import pytest

from semantic_layer.config import settings
from semantic_layer.graph.client import reset_graph
from semantic_layer.ingest import facts as facts_mod
from semantic_layer.ingest.facts import clean_facts, extract_facts_batch, fact_id, load_facts


class _Resp:
    def __init__(self, content):
        self.content = content


class _FakeModel:
    def __init__(self, content):
        self._content = content

    def invoke(self, _prompt):
        return _Resp(self._content)


def test_clean_facts_accepts_valid_triplet():
    out = clean_facts([
        {
            "subject": " Blackwell ",
            "predicate": " drove ",
            "object": " Data Center growth ",
            "confidence": 0.88,
            "valid_from": "FY2026-Q1",
            "valid_until": None,
        }
    ])

    assert out == [
        {
            "subject": "Blackwell",
            "predicate": "drove",
            "object": "Data Center growth",
            "text": "Blackwell / drove / Data Center growth",
            "confidence": 0.88,
            "valid_from": "FY2026-Q1",
            "valid_until": None,
        }
    ]


def test_clean_facts_rejects_malformed_rows():
    out = clean_facts([
        {"subject": "", "predicate": "drove", "object": "growth"},
        {"subject": "Blackwell", "predicate": "", "object": "growth"},
        {"subject": "Blackwell", "predicate": "drove", "object": ""},
        "Blackwell drove growth",
        None,
    ])

    assert out == []


def test_clean_facts_clamps_confidence():
    out = clean_facts([
        {"subject": "A", "predicate": "above", "object": "B", "confidence": 2},
        {"subject": "C", "predicate": "below", "object": "D", "confidence": -0.5},
        {"subject": "E", "predicate": "default", "object": "F"},
    ])

    assert [fact["confidence"] for fact in out] == [1.0, 0.0, 1.0]


def test_clean_facts_dedupes_duplicate_triplets_case_insensitively():
    out = clean_facts([
        {"subject": "Blackwell", "predicate": "drove", "object": "growth"},
        {"subject": "blackwell", "predicate": "DROVE", "object": "Growth"},
    ])

    assert out == [
        {
            "subject": "Blackwell",
            "predicate": "drove",
            "object": "growth",
            "text": "Blackwell / drove / growth",
            "confidence": 1.0,
            "valid_from": None,
            "valid_until": None,
        }
    ]


def test_fact_id_is_stable():
    first = fact_id("c1", "Blackwell", "drove", "growth")
    second = fact_id("c1", "Blackwell", "drove", "growth")
    other_chunk = fact_id("c2", "Blackwell", "drove", "growth")

    assert first == second
    assert first != other_chunk
    assert first.startswith("fact:")


@pytest.mark.neo4j
def test_load_facts_is_idempotent_and_links_chunk(neo4j_driver):
    reset_graph(neo4j_driver)
    with neo4j_driver.session(database=settings.neo4j_database) as session:
        session.run("CREATE (:Chunk {id: 'c1'})")
    facts = clean_facts([
        {"subject": "Blackwell", "predicate": "drove", "object": "growth"}
    ])

    assert load_facts(neo4j_driver, "c1", facts) == 1
    assert load_facts(neo4j_driver, "c1", facts) == 1

    with neo4j_driver.session(database=settings.neo4j_database) as session:
        row = session.run(
            """
            MATCH (:Chunk {id: 'c1'})-[r:HAS_FACT]->(f:Fact)
            RETURN count(r) AS links, count(DISTINCT f) AS facts, collect(f.text) AS texts
            """
        ).single()
    assert row["links"] == 1
    assert row["facts"] == 1
    assert row["texts"] == ["Blackwell / drove / growth"]


def test_extract_facts_batch_groups_per_chunk(monkeypatch):
    payload = json.dumps([
        [
            {
                "subject": "Blackwell",
                "predicate": "drove",
                "object": "growth",
                "confidence": 0.9,
            }
        ],
        [
            {"subject": "Data Center", "predicate": "grew", "object": "revenue"},
            {"subject": "", "predicate": "bad", "object": "row"},
        ],
    ])
    monkeypatch.setattr(facts_mod, "get_chat_model", lambda model=None: _FakeModel(payload))

    out = extract_facts_batch(["chunk one", "chunk two"])

    assert out == [
        [
            {
                "subject": "Blackwell",
                "predicate": "drove",
                "object": "growth",
                "text": "Blackwell / drove / growth",
                "confidence": 0.9,
                "valid_from": None,
                "valid_until": None,
            }
        ],
        [
            {
                "subject": "Data Center",
                "predicate": "grew",
                "object": "revenue",
                "text": "Data Center / grew / revenue",
                "confidence": 1.0,
                "valid_from": None,
                "valid_until": None,
            }
        ],
    ]


def test_extract_facts_batch_handles_bad_json(monkeypatch):
    monkeypatch.setattr(facts_mod, "get_chat_model", lambda model=None: _FakeModel("not json"))
    assert extract_facts_batch(["a", "b"]) == [[], []]


def test_extract_facts_batch_count_mismatch(monkeypatch):
    monkeypatch.setattr(facts_mod, "get_chat_model", lambda model=None: _FakeModel(json.dumps([])))
    assert extract_facts_batch(["a", "b"]) == [[], []]


def test_extract_facts_batch_empty_input(monkeypatch):
    monkeypatch.setattr(
        facts_mod,
        "get_chat_model",
        lambda model=None: (_ for _ in ()).throw(AssertionError("should not call model")),
    )
    assert extract_facts_batch([]) == []
