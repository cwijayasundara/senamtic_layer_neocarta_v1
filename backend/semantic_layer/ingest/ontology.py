"""Load the checked-in POLE+O ontology catalog into Neo4j."""

import json
from pathlib import Path
from typing import Any

from neo4j import Driver

from semantic_layer.config import settings

BASE_TYPES = {"Person", "Org", "Location", "Event", "Object"}
CATALOG_PATH = Path(__file__).with_name("ontology_catalog.json")


def load_catalog(path: Path = CATALOG_PATH) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    subtype_base_map(data)
    return data


def subtype_base_map(catalog: dict | None = None) -> dict[str, str]:
    if catalog is None:
        catalog = load_catalog()

    base_types = catalog.get("base_types")
    if not isinstance(base_types, list):
        raise ValueError("catalog base_types must be a list")
    if set(base_types) != BASE_TYPES or len(base_types) != len(BASE_TYPES):
        raise ValueError(f"catalog base_types must exactly match {sorted(BASE_TYPES)}")

    subtypes = catalog.get("subtypes")
    if not isinstance(subtypes, list):
        raise ValueError("catalog subtypes must be a list")

    mapping: dict[str, str] = {}
    for subtype in subtypes:
        if not isinstance(subtype, dict):
            raise ValueError("catalog subtype rows must be objects")

        name = subtype.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("catalog subtype missing name")
        if name in mapping:
            raise ValueError(f"duplicate subtype name: {name}")

        base_type = subtype.get("base_type")
        if not isinstance(base_type, str) or not base_type.strip() or base_type not in BASE_TYPES:
            raise ValueError(f"unknown base_type for subtype {name}: {base_type}")

        for field in ("domain", "description"):
            value = subtype.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"catalog subtype {name} missing {field}")

        mapping[name] = base_type

    return mapping


def load_ontology(driver: Driver, catalog: dict | None = None) -> int:
    if catalog is None:
        catalog = load_catalog()
    subtype_base_map(catalog)

    base_types = catalog["base_types"]
    subtypes = catalog["subtypes"]

    with driver.session(database=settings.neo4j_database) as session:
        session.run(
            """
            UNWIND $base_types AS base_type
            MERGE (:OntologyType {name: base_type})
            """,
            base_types=base_types,
        )
        session.run(
            """
            UNWIND $subtypes AS row
            MERGE (s:OntologySubtype {name: row.name})
            SET s.base_type = row.base_type,
                s.domain = row.domain,
                s.description = row.description
            MERGE (t:OntologyType {name: row.base_type})
            WITH s, t, row
            OPTIONAL MATCH (s)-[old_rel:SUBTYPE_OF]->(old:OntologyType)
            WHERE old.name <> row.base_type
            DELETE old_rel
            WITH s, t
            MERGE (s)-[:SUBTYPE_OF]->(t)
            """,
            subtypes=subtypes,
        )

    return len(subtypes)
