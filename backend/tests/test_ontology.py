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
    assert catalog["base_types"] == ["Person", "Org", "Location", "Event", "Object"]
    assert set(catalog["base_types"]) == BASE_TYPES
    assert catalog["subtypes"] == [
        {
            "name": "Product",
            "base_type": "Object",
            "domain": "nvidia_demo",
            "description": "A sellable product, product family, or product line.",
        },
        {
            "name": "ProductArchitecture",
            "base_type": "Object",
            "domain": "nvidia_demo",
            "description": "A hardware or platform architecture such as Blackwell.",
        },
        {
            "name": "Technology",
            "base_type": "Object",
            "domain": "nvidia_demo",
            "description": "A named technology, software stack, or technical capability.",
        },
        {
            "name": "Metric",
            "base_type": "Object",
            "domain": "nvidia_demo",
            "description": "A business, financial, operational, or telemetry measure.",
        },
        {
            "name": "DocumentArtifact",
            "base_type": "Object",
            "domain": "nvidia_demo",
            "description": "A report, filing, press release document, or other information artifact.",
        },
        {
            "name": "Customer",
            "base_type": "Org",
            "domain": "nvidia_demo",
            "description": "An organization buying products or consuming services.",
        },
        {
            "name": "Partner",
            "base_type": "Org",
            "domain": "nvidia_demo",
            "description": "An organization acting as a channel, supplier, or ecosystem partner.",
        },
        {
            "name": "BusinessUnit",
            "base_type": "Org",
            "domain": "nvidia_demo",
            "description": "An internal business group, segment, or operating unit.",
        },
        {
            "name": "Vendor",
            "base_type": "Org",
            "domain": "nvidia_demo",
            "description": "An external provider of products or services.",
        },
        {
            "name": "Region",
            "base_type": "Location",
            "domain": "nvidia_demo",
            "description": "A sales, support, or reporting region such as EMEA.",
        },
        {
            "name": "Country",
            "base_type": "Location",
            "domain": "nvidia_demo",
            "description": "A country or nation-state.",
        },
        {
            "name": "FiscalPeriod",
            "base_type": "Event",
            "domain": "nvidia_demo",
            "description": "A fiscal year, quarter, or reporting period.",
        },
        {
            "name": "PressRelease",
            "base_type": "Event",
            "domain": "nvidia_demo",
            "description": "A public announcement or earnings-related publication event.",
        },
        {
            "name": "SupportIncident",
            "base_type": "Event",
            "domain": "nvidia_demo",
            "description": "A support case, incident, or ticket.",
        },
        {
            "name": "SalesTransaction",
            "base_type": "Event",
            "domain": "nvidia_demo",
            "description": "A sale, order, booking, or revenue-generating transaction.",
        },
    ]


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
    assert count == 15
    with neo4j_driver.session(database=settings.neo4j_database) as session:
        base_count = session.run("MATCH (t:OntologyType) RETURN count(t) AS c").single()["c"]
        subtype_count = session.run("MATCH (s:OntologySubtype) RETURN count(s) AS c").single()["c"]
        subtype = session.run(
            """
            MATCH (s:OntologySubtype {name:'ProductArchitecture'})-[:SUBTYPE_OF]->(t:OntologyType)
            RETURN s.base_type AS base_type, t.name AS type_name
            """
        ).single()
    assert base_count == 5
    assert subtype_count == 15
    assert subtype["base_type"] == "Object"
    assert subtype["type_name"] == "Object"
