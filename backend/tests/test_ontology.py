import pytest

from semantic_layer.config import settings
from semantic_layer.graph.client import reset_graph
from semantic_layer.ingest.ontology import (
    BASE_TYPES,
    load_catalog,
    load_ontology,
    subtype_base_map,
)


def test_load_catalog_contains_fixed_poleo_base_types():
    catalog = load_catalog()
    assert set(catalog["base_types"]) == BASE_TYPES
    subtypes = {s["name"]: s["base_type"] for s in catalog["subtypes"]}
    assert subtypes["ProductArchitecture"] == "Object"
    assert subtypes["Customer"] == "Org"
    assert subtypes["Region"] == "Location"
    assert subtypes["FiscalPeriod"] == "Event"


def test_subtype_base_map_rejects_unknown_base_type():
    bad = {
        "base_types": ["Person", "Org", "Location", "Event", "Object"],
        "subtypes": [{"name": "Bad", "base_type": "Concept", "domain": "x", "description": "x"}],
    }
    with pytest.raises(ValueError, match="unknown base_type"):
        subtype_base_map(bad)


@pytest.mark.neo4j
def test_load_ontology_merges_base_types_and_subtypes(neo4j_driver):
    reset_graph(neo4j_driver)
    count = load_ontology(neo4j_driver)
    assert count >= 15
    with neo4j_driver.session(database=settings.neo4j_database) as session:
        base_count = session.run("MATCH (t:OntologyType) RETURN count(t) AS c").single()["c"]
        subtype = session.run(
            """
            MATCH (s:OntologySubtype {name:'ProductArchitecture'})-[:SUBTYPE_OF]->(t:OntologyType)
            RETURN s.base_type AS base_type, t.name AS type_name
            """
        ).single()
    assert base_count == 5
    assert subtype["base_type"] == "Object"
    assert subtype["type_name"] == "Object"
