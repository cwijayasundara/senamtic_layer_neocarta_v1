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


class _FakeSession:
    def __init__(self):
        self.params = None
        self.query = None
        self.chunk_exists = True

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def run(self, query, **params):
        self.query = query
        if "RETURN count(c) AS count" in query:
            return _FakeResult({"count": 1 if self.chunk_exists else 0})
        if " AS links" in query:
            return _FakeResult({"links": 0})
        self.params = params
        return _FakeResult({"loaded": len(params.get("rows", []))})


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def single(self):
        return self._row


class _FakeDriver:
    def __init__(self, chunk_exists=True):
        self.session_obj = _FakeSession()
        self.session_obj.chunk_exists = chunk_exists

    def session(self, database=None):
        return self.session_obj


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


def test_clean_facts_rejects_non_string_triplet_fields():
    out = clean_facts([
        {"subject": 123, "predicate": "p", "object": "o"},
        {"subject": "s", "predicate": ["p"], "object": "o"},
        {"subject": "s", "predicate": "p", "object": {"name": "o"}},
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


def test_load_facts_skips_malformed_direct_rows():
    driver = _FakeDriver()
    rows = [
        {
            "subject": "Blackwell",
            "predicate": "drove",
            "object": "growth",
            "text": "Blackwell / drove / growth",
            "confidence": 1.0,
            "valid_from": None,
            "valid_until": None,
        },
        {"subject": 123, "predicate": "p", "object": "o", "text": "bad", "confidence": 1.0},
        {"subject": "s", "predicate": ["p"], "object": "o", "text": "bad", "confidence": 1.0},
        {"subject": "s", "predicate": "p", "object": {"name": "o"}, "text": "bad", "confidence": 1.0},
    ]

    assert load_facts(driver, "c1", rows) == 1
    assert driver.session_obj.params["rows"][0]["text"] == "Blackwell / drove / growth"


def test_load_facts_normalizes_malformed_optional_direct_fields():
    driver = _FakeDriver()
    rows = [
        {
            "subject": " Blackwell ",
            "predicate": " drove ",
            "object": " growth ",
            "text": {"bad": "text"},
            "confidence": 0.7,
            "valid_from": ["FY2026-Q1"],
            "valid_until": {"quarter": "FY2026-Q2"},
        }
    ]

    assert load_facts(driver, "c1", rows) == 1
    loaded = driver.session_obj.params["rows"][0]
    assert loaded["subject"] == "Blackwell"
    assert loaded["predicate"] == "drove"
    assert loaded["object"] == "growth"
    assert loaded["subject_norm"] == "blackwell"
    assert loaded["object_norm"] == "growth"
    assert loaded["text"] == "Blackwell / drove / growth"
    assert loaded["valid_from"] is None
    assert loaded["valid_until"] is None


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


@pytest.mark.neo4j
def test_load_facts_replaces_stale_chunk_facts(neo4j_driver):
    reset_graph(neo4j_driver)
    with neo4j_driver.session(database=settings.neo4j_database) as session:
        session.run("CREATE (:Chunk {id: 'c1'})")

    assert load_facts(neo4j_driver, "c1", clean_facts([
        {"subject": "Blackwell", "predicate": "drove", "object": "growth"}
    ])) == 1
    assert load_facts(neo4j_driver, "c1", clean_facts([
        {"subject": "Blackwell", "predicate": "enabled", "object": "AI"}
    ])) == 1

    with neo4j_driver.session(database=settings.neo4j_database) as session:
        row = session.run(
            """
            MATCH (:Chunk {id: 'c1'})-[:HAS_FACT]->(f:Fact)
            RETURN count(f) AS linked, collect(f.text) AS texts
            """
        ).single()
        total = session.run("MATCH (f:Fact) RETURN count(f) AS count").single()["count"]

    assert row["linked"] == 1
    assert row["texts"] == ["Blackwell / enabled / AI"]
    assert total == 1


@pytest.mark.neo4j
def test_load_facts_empty_rows_clear_prior_chunk_facts(neo4j_driver):
    reset_graph(neo4j_driver)
    with neo4j_driver.session(database=settings.neo4j_database) as session:
        session.run("CREATE (:Chunk {id: 'c1'})")
    assert load_facts(neo4j_driver, "c1", clean_facts([
        {"subject": "Blackwell", "predicate": "drove", "object": "growth"}
    ])) == 1

    assert load_facts(neo4j_driver, "c1", []) == 0

    with neo4j_driver.session(database=settings.neo4j_database) as session:
        row = session.run(
            """
            MATCH (:Chunk {id: 'c1'})
            OPTIONAL MATCH (:Chunk {id: 'c1'})-[r:HAS_FACT]->(f:Fact)
            RETURN count(r) AS links, count(f) AS linked_facts
            """
        ).single()
        total = session.run("MATCH (f:Fact) RETURN count(f) AS count").single()["count"]

    assert row["links"] == 0
    assert row["linked_facts"] == 0
    assert total == 0


@pytest.mark.neo4j
def test_load_facts_missing_chunk_returns_zero_and_creates_no_facts(neo4j_driver):
    reset_graph(neo4j_driver)
    facts = clean_facts([
        {"subject": "Blackwell", "predicate": "drove", "object": "growth"}
    ])

    assert load_facts(neo4j_driver, "c-missing", facts) == 0

    with neo4j_driver.session(database=settings.neo4j_database) as session:
        count = session.run("MATCH (f:Fact) RETURN count(f) AS count").single()["count"]
    assert count == 0


def test_link_fact_anchor_query_uses_scoped_subqueries():
    driver = _FakeDriver()
    facts_mod._link_fact_anchor_count(driver, "subject_norm", "SUBJECT_REFERS_TO")
    query = driver.session_obj.query

    assert "CALL (f) {" in query
    assert "CALL {\n                WITH f" not in query


@pytest.mark.neo4j
def test_link_facts_to_entities_and_values(neo4j_driver):
    from semantic_layer.ingest.facts import link_facts

    reset_graph(neo4j_driver)
    with neo4j_driver.session(database=settings.neo4j_database) as session:
        session.run(
            """
            CREATE (:Chunk {id:'c1', text:'Blackwell drove Data Center growth.'})
            CREATE (:Entity {norm:'blackwell', name:'Blackwell', label:'Object'})
            CREATE (:Value {norm:'data center', name:'Data Center'})
            """
        )
    facts = clean_facts([{"subject": "Blackwell", "predicate": "drove", "object": "Data Center"}])
    load_facts(neo4j_driver, "c1", facts)

    counts = link_facts(neo4j_driver)

    assert counts["subject_links"] == 1
    assert counts["object_links"] == 1


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
