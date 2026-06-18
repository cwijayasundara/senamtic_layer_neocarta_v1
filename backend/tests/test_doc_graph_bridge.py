"""Token-overlap bridging of document Entities to catalog Values.

Exact-norm equality misses real correspondences whose surface forms differ
('NVIDIA Blackwell GPUs' vs 'Blackwell'). _token_match adds whole-word overlap
while guarding against substring false positives ('Indiana Jones' vs 'India')
and short codes ('us', 'q1')."""

import pytest

from semantic_layer.config import settings
from semantic_layer.graph.client import reset_graph
from semantic_layer.ingest.doc_loader import load_document
from semantic_layer.ingest.doc_graph import (
    _token_match,
    load_entities,
    bridge_entities_to_values,
)


# --- pure predicate: no database ---------------------------------------------

@pytest.mark.parametrize("entity, value", [
    ("blackwell", "blackwell"),                       # exact
    ("nvidia blackwell", "blackwell"),                # value is a trailing token
    ("blackwell ultra gpus", "blackwell"),            # value is a leading token
    ("nvidia grace blackwell platform", "grace"),     # value is an interior token
    ("gaming and ai pc", "gaming"),                   # token amid filler
    ("research and development", "research"),         # token before a stopword
    ("nvidia data center group", "data center"),      # contiguous multi-word value
])
def test_token_match_accepts_whole_word_overlap(entity, value):
    assert _token_match(entity, value) is True


@pytest.mark.parametrize("entity, value", [
    ("indiana jones and the great circle", "india"),  # substring, not a whole word
    ("data flows to the center", "data center"),       # multi-word, not contiguous
    ("revenue in the us", "us"),                       # short code guard (<4 chars)
    ("results for q1", "q1"),                          # short code guard
    ("foo and bar", "and"),                            # stopword token
    ("colette kress", "blackwell"),                    # no overlap at all
])
def test_token_match_rejects_false_positives(entity, value):
    assert _token_match(entity, value) is False


# --- integration: real near-miss data ----------------------------------------

@pytest.mark.neo4j
def test_bridge_links_token_overlap_but_not_substring(neo4j_driver):
    reset_graph(neo4j_driver)
    with neo4j_driver.session(database=settings.neo4j_database) as s:
        s.run(
            """
            MERGE (v1:Value {norm:'blackwell'}) SET v1.name='Blackwell'
            MERGE (v2:Value {norm:'india'})     SET v2.name='India'
            """
        )
    load_document(neo4j_driver, {
        "doc_id": "doc:pr", "title": "pr", "path": "/tmp/pr.pdf", "num_pages": 1,
        "chunks": [{"chunk_id": "doc:pr:chunk:0", "doc_id": "doc:pr", "ordinal": 0,
                    "text": "x"}],
    })
    load_entities(neo4j_driver, "doc:pr:chunk:0", [
        {"name": "NVIDIA Blackwell GPUs", "label": "Object"},
        {"name": "Indiana Jones and the Great Circle", "label": "Object"},
    ])
    bridge_entities_to_values(neo4j_driver)

    with neo4j_driver.session(database=settings.neo4j_database) as s:
        linked = {
            r["en"]: r["vn"] for r in s.run(
                "MATCH (e:Entity)-[:REFERS_TO]->(v:Value) RETURN e.norm AS en, v.norm AS vn"
            )
        }
    assert linked.get("nvidia blackwell gpus") == "blackwell"   # token overlap bridged
    assert "indiana jones and the great circle" not in linked   # substring did NOT
