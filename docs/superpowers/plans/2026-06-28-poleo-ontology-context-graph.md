# POLE+O Ontology Context Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small POLE+O ontology layer, typed entity extraction, and searchable Fact nodes to improve planning and document grounding without replacing the existing NeoCarta/value graph.

**Architecture:** Keep `Entity.label` as the POLE+O base type for compatibility, then add ontology catalog nodes and `Entity-[:INSTANCE_OF]->OntologySubtype` edges. Facts are independent atomic triplet nodes linked to chunks and best-effort matched back to entities/values. The ingest pipeline loads deterministic ontology data even in `with_llm=False`, and only runs typed entity/fact extraction inside the existing LLM stage.

**Tech Stack:** Python 3.11, Neo4j driver/Cypher, LangChain chat model wrappers, OpenAI embeddings, pytest.

---

## File Structure

- Create `backend/semantic_layer/ingest/ontology_catalog.json`: checked-in subtype vocabulary.
- Create `backend/semantic_layer/ingest/ontology.py`: catalog validation/loading and entity subtype linking helpers.
- Modify `backend/semantic_layer/ingest/entities.py`: typed extraction schema, backwards-compatible cleaning, confidence threshold.
- Modify `backend/semantic_layer/ingest/doc_graph.py`: preserve base labels and load valid subtype edges.
- Create `backend/semantic_layer/ingest/facts.py`: Fact cleaning, id generation, loading, linking, and extraction.
- Modify `backend/semantic_layer/ingest/embeddings.py`: embed `Fact.text` and create `fact_embeddings`.
- Modify `backend/semantic_layer/ingest/pipeline.py`: load ontology catalog deterministically, then load typed entities, Facts, and fact embeddings in LLM stages.
- Modify `backend/semantic_layer/agent/graph_tools.py`: add `search_facts`, include facts in `neighbors`.
- Modify `backend/semantic_layer/web/graph_api.py`: expose entity subtype in graph node payloads.
- Test with focused additions in `backend/tests/test_ontology.py`, `backend/tests/test_entities_batch.py`, `backend/tests/test_doc_graph_bridge.py`, `backend/tests/test_facts.py`, `backend/tests/test_agent_graph_tools.py`, `backend/tests/test_pipeline.py`, and `backend/tests/test_web_graph_api.py`.

## Task 1: Ontology Catalog Loader

**Files:**
- Create: `backend/semantic_layer/ingest/ontology_catalog.json`
- Create: `backend/semantic_layer/ingest/ontology.py`
- Test: `backend/tests/test_ontology.py`

- [ ] **Step 1: Write failing tests for catalog validation and loading**

Add `backend/tests/test_ontology.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd backend
PYTHONPATH=. uv run pytest tests/test_ontology.py -q
```

Expected: FAIL because `semantic_layer.ingest.ontology` does not exist.

- [ ] **Step 3: Add catalog JSON**

Create `backend/semantic_layer/ingest/ontology_catalog.json`:

```json
{
  "base_types": ["Person", "Org", "Location", "Event", "Object"],
  "subtypes": [
    {
      "name": "Product",
      "base_type": "Object",
      "domain": "nvidia_demo",
      "description": "A sellable product, product family, or product line."
    },
    {
      "name": "ProductArchitecture",
      "base_type": "Object",
      "domain": "nvidia_demo",
      "description": "A hardware or platform architecture such as Blackwell."
    },
    {
      "name": "Technology",
      "base_type": "Object",
      "domain": "nvidia_demo",
      "description": "A named technology, software stack, or technical capability."
    },
    {
      "name": "Metric",
      "base_type": "Object",
      "domain": "nvidia_demo",
      "description": "A business, financial, operational, or telemetry measure."
    },
    {
      "name": "DocumentArtifact",
      "base_type": "Object",
      "domain": "nvidia_demo",
      "description": "A report, filing, press release document, or other information artifact."
    },
    {
      "name": "Customer",
      "base_type": "Org",
      "domain": "nvidia_demo",
      "description": "An organization buying products or consuming services."
    },
    {
      "name": "Partner",
      "base_type": "Org",
      "domain": "nvidia_demo",
      "description": "An organization acting as a channel, supplier, or ecosystem partner."
    },
    {
      "name": "BusinessUnit",
      "base_type": "Org",
      "domain": "nvidia_demo",
      "description": "An internal business group, segment, or operating unit."
    },
    {
      "name": "Vendor",
      "base_type": "Org",
      "domain": "nvidia_demo",
      "description": "An external provider of products or services."
    },
    {
      "name": "Region",
      "base_type": "Location",
      "domain": "nvidia_demo",
      "description": "A sales, support, or reporting region such as EMEA."
    },
    {
      "name": "Country",
      "base_type": "Location",
      "domain": "nvidia_demo",
      "description": "A country or nation-state."
    },
    {
      "name": "FiscalPeriod",
      "base_type": "Event",
      "domain": "nvidia_demo",
      "description": "A fiscal year, quarter, or reporting period."
    },
    {
      "name": "PressRelease",
      "base_type": "Event",
      "domain": "nvidia_demo",
      "description": "A public announcement or earnings-related publication event."
    },
    {
      "name": "SupportIncident",
      "base_type": "Event",
      "domain": "nvidia_demo",
      "description": "A support case, incident, or ticket."
    },
    {
      "name": "SalesTransaction",
      "base_type": "Event",
      "domain": "nvidia_demo",
      "description": "A sale, order, booking, or revenue-generating transaction."
    }
  ]
}
```

- [ ] **Step 4: Implement ontology loader**

Create `backend/semantic_layer/ingest/ontology.py`:

```python
"""Load the small POLE+O subtype ontology used by document extraction."""

import json
from pathlib import Path
from typing import Any

from neo4j import Driver

from semantic_layer.config import settings

BASE_TYPES = {"Person", "Org", "Location", "Event", "Object"}
CATALOG_PATH = Path(__file__).with_name("ontology_catalog.json")


def load_catalog(path: Path = CATALOG_PATH) -> dict[str, Any]:
    """Read and validate the checked-in ontology catalog."""
    catalog = json.loads(path.read_text())
    subtype_base_map(catalog)
    return catalog


def subtype_base_map(catalog: dict[str, Any] | None = None) -> dict[str, str]:
    """Return subtype -> base type, raising on invalid catalog entries."""
    data = catalog if catalog is not None else json.loads(CATALOG_PATH.read_text())
    base_types = set(data.get("base_types") or [])
    if base_types != BASE_TYPES:
        raise ValueError(f"base_types must be exactly {sorted(BASE_TYPES)}")
    out: dict[str, str] = {}
    for row in data.get("subtypes") or []:
        name = (row.get("name") or "").strip()
        base = (row.get("base_type") or "").strip()
        if not name:
            raise ValueError("subtype name is required")
        if base not in BASE_TYPES:
            raise ValueError(f"unknown base_type for subtype {name}: {base}")
        if name in out:
            raise ValueError(f"duplicate subtype: {name}")
        out[name] = base
    return out


def load_ontology(driver: Driver, catalog: dict[str, Any] | None = None) -> int:
    """MERGE OntologyType/OntologySubtype nodes and SUBTYPE_OF edges.

    Returns the number of subtype rows loaded. Base type count is fixed at five.
    """
    data = catalog or load_catalog()
    rows = [
        {
            "name": s["name"],
            "base_type": s["base_type"],
            "domain": s.get("domain", ""),
            "description": s.get("description", ""),
        }
        for s in data["subtypes"]
    ]
    with driver.session(database=settings.neo4j_database) as session:
        session.run(
            """
            UNWIND $base_types AS name
            MERGE (:OntologyType {name: name})
            """,
            base_types=sorted(BASE_TYPES),
        )
        session.run(
            """
            UNWIND $rows AS row
            MATCH (t:OntologyType {name: row.base_type})
            MERGE (s:OntologySubtype {name: row.name})
              SET s.base_type = row.base_type,
                  s.domain = row.domain,
                  s.description = row.description
            MERGE (s)-[:SUBTYPE_OF]->(t)
            """,
            rows=rows,
        )
    return len(rows)
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
cd backend
PYTHONPATH=. uv run pytest tests/test_ontology.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/semantic_layer/ingest/ontology_catalog.json backend/semantic_layer/ingest/ontology.py backend/tests/test_ontology.py
git commit -m "feat: add POLE ontology catalog"
```

## Task 2: Typed Entity Cleaning and Prompt

**Files:**
- Modify: `backend/semantic_layer/ingest/entities.py`
- Modify: `backend/tests/test_entities.py`
- Modify: `backend/tests/test_entities_batch.py`

- [ ] **Step 1: Write failing tests for typed entity cleaning**

Append to `backend/tests/test_entities_batch.py`:

```python
from semantic_layer.ingest.entities import _clean_entities


def test_clean_entities_accepts_typed_entity():
    rows = _clean_entities([
        {
            "name": "Blackwell",
            "base_type": "Object",
            "subtype": "ProductArchitecture",
            "confidence": 0.91,
            "evidence": "Blackwell architecture drove Data Center growth.",
        }
    ])
    assert rows == [{
        "name": "Blackwell",
        "label": "Object",
        "base_type": "Object",
        "subtype": "ProductArchitecture",
        "confidence": 0.91,
        "evidence": "Blackwell architecture drove Data Center growth.",
    }]


def test_clean_entities_degrades_low_confidence_subtype():
    rows = _clean_entities([
        {"name": "Blackwell", "base_type": "Object", "subtype": "ProductArchitecture", "confidence": 0.79}
    ])
    assert rows == [{
        "name": "Blackwell",
        "label": "Object",
        "base_type": "Object",
        "subtype": None,
        "confidence": 0.79,
        "evidence": "",
    }]


def test_clean_entities_accepts_legacy_label_shape():
    rows = _clean_entities([{"name": "NVIDIA", "label": "Org"}])
    assert rows == [{
        "name": "NVIDIA",
        "label": "Org",
        "base_type": "Org",
        "subtype": None,
        "confidence": 1.0,
        "evidence": "",
    }]
```

Update `backend/tests/test_entities.py` live OpenAI assertion:

```python
    assert all(e["label"] in POLE_LABELS for e in ents)
    assert all(e["base_type"] in POLE_LABELS for e in ents)
    assert all("subtype" in e for e in ents)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd backend
PYTHONPATH=. uv run pytest tests/test_entities_batch.py tests/test_entities.py -q
```

Expected: FAIL because `_clean_entities` does not expose typed fields yet.

- [ ] **Step 3: Implement typed cleaning and prompt vocabulary**

Modify `backend/semantic_layer/ingest/entities.py`:

```python
"""Extract POLE+O entities (Person, Org, Location, Event, Object) from text via LLM."""

import json

from semantic_layer.ingest.llm import get_chat_model
from semantic_layer.ingest.ontology import subtype_base_map

POLE_LABELS = {"Person", "Org", "Location", "Event", "Object"}
SUBTYPE_CONFIDENCE_THRESHOLD = 0.80


def _subtype_prompt() -> str:
    mapping = subtype_base_map()
    grouped: dict[str, list[str]] = {label: [] for label in sorted(POLE_LABELS)}
    for subtype, base in sorted(mapping.items()):
        grouped[base].append(subtype)
    return "; ".join(
        f"{base}: {', '.join(names) if names else 'none'}"
        for base, names in grouped.items()
    )


_PROMPT = (
    "Extract named entities from the text. Return ONLY a JSON array of objects "
    'with keys "name", "base_type", "subtype", "confidence", and "evidence". '
    "base_type must be one of: Person, Org, Location, Event, Object. "
    "subtype must be one of the allowed subtypes for that base_type, or null when no "
    "subtype fits. confidence is a number from 0 to 1 for the subtype assignment. "
    "evidence is a short quote or phrase from the text. Allowed subtypes: {subtypes}. "
    "Deduplicate by name. Text:\n\n{text}"
)


def extract_entities(text: str) -> list[dict]:
    model = get_chat_model()
    resp = model.invoke(_PROMPT.format(text=text[:6000], subtypes=_subtype_prompt()))
    content = resp.content if hasattr(resp, "content") else str(resp)
    content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        return []
    return _clean_entities(raw)


_BATCH_PROMPT = (
    "Extract named entities from EACH numbered text below. Return ONLY a JSON array "
    "with one element per text, in the same order. Each element is an array of objects "
    'with keys "name", "base_type", "subtype", "confidence", and "evidence"; '
    "base_type must be one of: Person, Org, Location, Event, Object. subtype must be "
    "one of the allowed subtypes for that base_type, or null when no subtype fits. "
    "confidence is a number from 0 to 1 for the subtype assignment. evidence is a short "
    "quote or phrase from the text. Deduplicate by name within each text. Use an empty "
    "array for a text with no entities. Return exactly {n} elements. Allowed subtypes: "
    "{subtypes}.\n\n{body}"
)


def _coerce_confidence(value) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, conf))


def _clean_entities(raw: list) -> list[dict]:
    out, seen = [], set()
    subtype_map = subtype_base_map()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip()
        base_type = (item.get("base_type") or item.get("label") or "").strip().capitalize()
        if base_type == "Organization":
            base_type = "Org"
        confidence = _coerce_confidence(item.get("confidence", 1.0))
        subtype = item.get("subtype")
        subtype = str(subtype).strip() if subtype not in (None, "") else None
        if (
            subtype
            and (subtype_map.get(subtype) != base_type or confidence < SUBTYPE_CONFIDENCE_THRESHOLD)
        ):
            subtype = None
        if name and base_type in POLE_LABELS and name.lower() not in seen:
            seen.add(name.lower())
            out.append({
                "name": name,
                "label": base_type,
                "base_type": base_type,
                "subtype": subtype,
                "confidence": confidence,
                "evidence": (item.get("evidence") or "").strip(),
            })
    return out


def extract_entities_batch(texts: list[str]) -> list[list[dict]]:
    """Extract typed POLE+O entities for many chunks in ONE LLM call."""
    if not texts:
        return []
    body = "\n\n".join(f"[{i}] {t[:6000]}" for i, t in enumerate(texts))
    model = get_chat_model()
    resp = model.invoke(_BATCH_PROMPT.format(n=len(texts), body=body, subtypes=_subtype_prompt()))
    content = resp.content if hasattr(resp, "content") else str(resp)
    content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        groups = json.loads(content)
    except json.JSONDecodeError:
        return [[] for _ in texts]
    if not isinstance(groups, list) or len(groups) != len(texts):
        return [[] for _ in texts]
    return [_clean_entities(g) for g in groups]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd backend
PYTHONPATH=. uv run pytest tests/test_entities_batch.py tests/test_entities.py -q
```

Expected: PASS, except `test_extract_entities_finds_nvidia_org` may SKIP if no OpenAI key is configured.

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/ingest/entities.py backend/tests/test_entities.py backend/tests/test_entities_batch.py
git commit -m "feat: extract typed POLE entities"
```

## Task 3: Load Entity Subtype Edges

**Files:**
- Modify: `backend/semantic_layer/ingest/doc_graph.py`
- Modify: `backend/tests/test_doc_graph_bridge.py`
- Test: `backend/tests/test_ontology.py`

- [ ] **Step 1: Write failing subtype-loading integration test**

Append to `backend/tests/test_doc_graph_bridge.py`:

```python
@pytest.mark.neo4j
def test_load_entities_links_valid_subtype(neo4j_driver):
    from semantic_layer.ingest.ontology import load_ontology

    reset_graph(neo4j_driver)
    load_ontology(neo4j_driver)
    load_document(neo4j_driver, {
        "doc_id": "doc:pr", "title": "pr", "path": "/tmp/pr.pdf", "num_pages": 1,
        "chunks": [{"chunk_id": "doc:pr:chunk:0", "doc_id": "doc:pr", "ordinal": 0,
                    "text": "Blackwell drove growth."}],
    })
    load_entities(neo4j_driver, "doc:pr:chunk:0", [
        {
            "name": "Blackwell",
            "label": "Object",
            "base_type": "Object",
            "subtype": "ProductArchitecture",
            "confidence": 0.91,
            "evidence": "Blackwell",
        }
    ])

    with neo4j_driver.session(database=settings.neo4j_database) as session:
        row = session.run(
            """
            MATCH (e:Entity {norm:'blackwell'})-[:INSTANCE_OF]->(s:OntologySubtype)
            RETURN e.label AS label, e.confidence AS confidence, s.name AS subtype
            """
        ).single()
    assert row["label"] == "Object"
    assert row["subtype"] == "ProductArchitecture"
    assert row["confidence"] == 0.91
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd backend
PYTHONPATH=. uv run pytest tests/test_doc_graph_bridge.py::test_load_entities_links_valid_subtype -q
```

Expected: FAIL because `load_entities` does not create `INSTANCE_OF`.

- [ ] **Step 3: Update `load_entities` to write subtype edges**

Modify `backend/semantic_layer/ingest/doc_graph.py` `load_entities`:

```python
def load_entities(driver: Driver, chunk_id: str, entities: list[dict]) -> None:
    """MERGE Entity {norm} nodes and Chunk-[:MENTIONS]->Entity edges for one chunk."""
    rows = [
        {
            "name": e["name"],
            "label": e.get("base_type") or e.get("label"),
            "norm": norm(e["name"]),
            "subtype": e.get("subtype"),
            "confidence": e.get("confidence", 1.0),
            "evidence": e.get("evidence", ""),
        }
        for e in entities if (e.get("name") or "").strip()
    ]
    if not rows:
        return
    with driver.session(database=settings.neo4j_database) as session:
        session.run(
            """
            MATCH (c:Chunk {id: $chunk_id})
            UNWIND $rows AS row
            MERGE (e:Entity {norm: row.norm})
              ON CREATE SET e.name = row.name
              SET e.label = row.label,
                  e.confidence = row.confidence,
                  e.evidence = row.evidence
            MERGE (c)-[:MENTIONS]->(e)
            WITH e, row
            WHERE row.subtype IS NOT NULL
            MATCH (s:OntologySubtype {name: row.subtype, base_type: row.label})
            MERGE (e)-[:INSTANCE_OF]->(s)
            """,
            chunk_id=chunk_id, rows=rows,
        )
```

- [ ] **Step 4: Run bridge tests**

Run:

```bash
cd backend
PYTHONPATH=. uv run pytest tests/test_doc_graph_bridge.py tests/test_ontology.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/ingest/doc_graph.py backend/tests/test_doc_graph_bridge.py
git commit -m "feat: link entities to ontology subtypes"
```

## Task 4: Pipeline Loads Ontology Catalog

**Files:**
- Modify: `backend/semantic_layer/ingest/pipeline.py`
- Modify: `backend/tests/test_pipeline.py`

- [ ] **Step 1: Write failing pipeline test**

Append to `backend/tests/test_pipeline.py`:

```python
def test_run_ingest_loads_ontology_without_llm(monkeypatch):
    calls = []

    class FakeDriver:
        def close(self):
            calls.append("close")

    monkeypatch.setattr("semantic_layer.ingest.pipeline.get_driver", lambda: FakeDriver())
    monkeypatch.setattr("semantic_layer.ingest.pipeline.reset_graph", lambda driver: calls.append("reset"))
    monkeypatch.setattr("semantic_layer.ingest.pipeline.extract_postgres", lambda *a, **k: type("B", (), {
        "databases": [], "schemas": [], "tables": [], "columns": [], "has_schema": [],
        "has_table": [], "has_column": [], "references": []
    })())
    monkeypatch.setattr("semantic_layer.ingest.pipeline.extract_sqlite", lambda *a, **k: type("B", (), {
        "databases": [], "schemas": [], "tables": [], "columns": [], "has_schema": [],
        "has_table": [], "has_column": [], "references": []
    })())
    monkeypatch.setattr("semantic_layer.ingest.pipeline.extract_all_apis", lambda *a, **k: [])
    monkeypatch.setattr("semantic_layer.ingest.pipeline._scale_bundles", lambda: [])
    monkeypatch.setattr("semantic_layer.ingest.pipeline.load_bundle", lambda *a, **k: calls.append("bundle"))
    monkeypatch.setattr("semantic_layer.ingest.pipeline.index_values", lambda *a, **k: 0)
    monkeypatch.setattr("semantic_layer.ingest.pipeline.index_periods", lambda *a, **k: 0)
    monkeypatch.setattr("semantic_layer.ingest.pipeline.bridge_sources", lambda *a, **k: 0)
    monkeypatch.setattr("semantic_layer.ingest.pipeline.index_query_log", lambda *a, **k: 0)
    monkeypatch.setattr("semantic_layer.ingest.pipeline.parse_document", lambda *a, **k: {})
    monkeypatch.setattr("semantic_layer.ingest.pipeline.load_document", lambda *a, **k: None)
    monkeypatch.setattr("semantic_layer.ingest.pipeline.link_document_period", lambda *a, **k: None)
    monkeypatch.setattr("semantic_layer.ingest.pipeline.extract_period", lambda *a, **k: None)
    monkeypatch.setattr("semantic_layer.ingest.pipeline.file_content_hash", lambda *a, **k: "hash")
    monkeypatch.setattr("semantic_layer.ingest.pipeline.document_unchanged", lambda *a, **k: False)
    monkeypatch.setattr("semantic_layer.ingest.pipeline.load_ontology", lambda *a, **k: calls.append("ontology") or 15)
    monkeypatch.setattr("semantic_layer.ingest.pipeline.Path.glob", lambda *a, **k: [])

    counts = __import__("semantic_layer.ingest.pipeline", fromlist=["run_ingest"]).run_ingest(
        with_llm=False, reset=True
    )

    assert "ontology" in calls
    assert counts["ontology_subtypes"] == 15
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd backend
PYTHONPATH=. uv run pytest tests/test_pipeline.py::test_run_ingest_loads_ontology_without_llm -q
```

Expected: FAIL because pipeline does not import or call `load_ontology`.

- [ ] **Step 3: Wire ontology loading into pipeline**

Modify imports in `backend/semantic_layer/ingest/pipeline.py`:

```python
from semantic_layer.ingest.ontology import load_ontology
```

Add after document ingest and before `if with_llm:`:

```python
        counts["ontology_subtypes"] = load_ontology(driver)
```

- [ ] **Step 4: Run pipeline tests**

Run:

```bash
cd backend
PYTHONPATH=. uv run pytest tests/test_pipeline.py tests/test_ontology.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/ingest/pipeline.py backend/tests/test_pipeline.py
git commit -m "feat: load ontology during ingest"
```

## Task 5: Fact Cleaning, IDs, and Loading

**Files:**
- Create: `backend/semantic_layer/ingest/facts.py`
- Test: `backend/tests/test_facts.py`

- [ ] **Step 1: Write failing tests for fact helpers**

Create `backend/tests/test_facts.py`:

```python
import pytest

from semantic_layer.config import settings
from semantic_layer.graph.client import reset_graph
from semantic_layer.ingest.facts import clean_facts, fact_id, load_facts


def test_clean_facts_accepts_valid_triplet():
    rows = clean_facts([
        {
            "subject": "Blackwell",
            "predicate": "drove",
            "object": "Data Center growth",
            "confidence": 0.88,
            "valid_from": "FY2026-Q1",
            "valid_until": None,
        }
    ])
    assert rows == [{
        "subject": "Blackwell",
        "predicate": "drove",
        "object": "Data Center growth",
        "text": "Blackwell / drove / Data Center growth",
        "confidence": 0.88,
        "valid_from": "FY2026-Q1",
        "valid_until": None,
    }]


def test_clean_facts_rejects_malformed_rows():
    assert clean_facts([
        {"subject": "", "predicate": "drove", "object": "growth"},
        {"subject": "Blackwell", "predicate": "", "object": "growth"},
        {"subject": "Blackwell", "predicate": "drove", "object": ""},
        "not a dict",
    ]) == []


def test_fact_id_is_stable():
    assert fact_id("chunk:1", "Blackwell", "drove", "growth") == fact_id(
        "chunk:1", "Blackwell", "drove", "growth"
    )
    assert fact_id("chunk:2", "Blackwell", "drove", "growth") != fact_id(
        "chunk:1", "Blackwell", "drove", "growth"
    )


@pytest.mark.neo4j
def test_load_facts_is_idempotent_and_links_chunk(neo4j_driver):
    reset_graph(neo4j_driver)
    with neo4j_driver.session(database=settings.neo4j_database) as session:
        session.run("CREATE (:Chunk {id:'c1', text:'Blackwell drove growth.'})")
    facts = clean_facts([{"subject": "Blackwell", "predicate": "drove", "object": "growth"}])

    assert load_facts(neo4j_driver, "c1", facts) == 1
    assert load_facts(neo4j_driver, "c1", facts) == 1

    with neo4j_driver.session(database=settings.neo4j_database) as session:
        row = session.run(
            """
            MATCH (:Chunk {id:'c1'})-[:HAS_FACT]->(f:Fact)
            RETURN count(f) AS count, collect(f.text) AS texts
            """
        ).single()
    assert row["count"] == 1
    assert row["texts"] == ["Blackwell / drove / growth"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd backend
PYTHONPATH=. uv run pytest tests/test_facts.py -q
```

Expected: FAIL because `semantic_layer.ingest.facts` does not exist.

- [ ] **Step 3: Implement fact helpers**

Create `backend/semantic_layer/ingest/facts.py`:

```python
"""Extract and load atomic Fact triplets from document chunks."""

import hashlib
import json

from neo4j import Driver

from semantic_layer.config import settings
from semantic_layer.ingest.llm import get_chat_model


_FACT_PROMPT = (
    "Extract atomic factual claims from EACH numbered text below. Return ONLY a JSON array "
    "with one element per text, in the same order. Each element is an array of objects "
    'with keys "subject", "predicate", "object", "confidence", "valid_from", and '
    '"valid_until". Each row must contain one concise factual triplet. Use an empty array '
    "for text with no clear factual claims. Return exactly {n} elements.\n\n{body}"
)


def _confidence(value) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, conf))


def clean_facts(raw: list) -> list[dict]:
    out = []
    seen = set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        subject = (item.get("subject") or "").strip()
        predicate = (item.get("predicate") or "").strip()
        obj = (item.get("object") or "").strip()
        if not subject or not predicate or not obj:
            continue
        key = (subject.lower(), predicate.lower(), obj.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "text": f"{subject} / {predicate} / {obj}",
            "confidence": _confidence(item.get("confidence", 1.0)),
            "valid_from": item.get("valid_from"),
            "valid_until": item.get("valid_until"),
        })
    return out


def fact_id(chunk_id: str, subject: str, predicate: str, obj: str) -> str:
    raw = "\x1f".join([chunk_id, subject.lower(), predicate.lower(), obj.lower()])
    return "fact:" + hashlib.sha256(raw.encode()).hexdigest()[:32]


def load_facts(driver: Driver, chunk_id: str, facts: list[dict]) -> int:
    rows = [
        {
            "id": fact_id(chunk_id, f["subject"], f["predicate"], f["object"]),
            "source_chunk_id": chunk_id,
            **f,
        }
        for f in facts
    ]
    if not rows:
        return 0
    with driver.session(database=settings.neo4j_database) as session:
        session.run(
            """
            MATCH (c:Chunk {id: $chunk_id})
            UNWIND $rows AS row
            MERGE (f:Fact {id: row.id})
              SET f.subject = row.subject,
                  f.predicate = row.predicate,
                  f.object = row.object,
                  f.text = row.text,
                  f.confidence = row.confidence,
                  f.source_chunk_id = row.source_chunk_id,
                  f.valid_from = row.valid_from,
                  f.valid_until = row.valid_until
            MERGE (c)-[:HAS_FACT]->(f)
            """,
            chunk_id=chunk_id,
            rows=rows,
        )
    return len(rows)


def extract_facts_batch(texts: list[str]) -> list[list[dict]]:
    if not texts:
        return []
    body = "\n\n".join(f"[{i}] {t[:6000]}" for i, t in enumerate(texts))
    model = get_chat_model()
    resp = model.invoke(_FACT_PROMPT.format(n=len(texts), body=body))
    content = resp.content if hasattr(resp, "content") else str(resp)
    content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        groups = json.loads(content)
    except json.JSONDecodeError:
        return [[] for _ in texts]
    if not isinstance(groups, list) or len(groups) != len(texts):
        return [[] for _ in texts]
    return [clean_facts(g) for g in groups]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd backend
PYTHONPATH=. uv run pytest tests/test_facts.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/ingest/facts.py backend/tests/test_facts.py
git commit -m "feat: add fact triplet loader"
```

## Task 6: Link Facts to Entities and Values

**Files:**
- Modify: `backend/semantic_layer/ingest/facts.py`
- Modify: `backend/tests/test_facts.py`

- [ ] **Step 1: Write failing linking test**

Append to `backend/tests/test_facts.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd backend
PYTHONPATH=. uv run pytest tests/test_facts.py::test_link_facts_to_entities_and_values -q
```

Expected: FAIL because `link_facts` does not exist.

- [ ] **Step 3: Implement fact linking**

Add to `backend/semantic_layer/ingest/facts.py`:

```python
from semantic_layer.ingest.value_indexer import norm
```

Update `load_facts` row construction:

```python
            "subject_norm": norm(f["subject"]),
            "object_norm": norm(f["object"]),
```

Add these properties in Cypher:

```cypher
                  f.subject_norm = row.subject_norm,
                  f.object_norm = row.object_norm,
```

Add function:

```python
def link_facts(driver: Driver) -> dict[str, int]:
    """Best-effort link Fact subjects/objects to Entity or Value nodes by norm."""
    with driver.session(database=settings.neo4j_database) as session:
        subject_entities = session.run(
            """
            MATCH (f:Fact), (e:Entity {norm: f.subject_norm})
            MERGE (f)-[:SUBJECT_REFERS_TO]->(e)
            RETURN count(*) AS c
            """
        ).single()["c"]
        subject_values = session.run(
            """
            MATCH (f:Fact), (v:Value {norm: f.subject_norm})
            MERGE (f)-[:SUBJECT_REFERS_TO]->(v)
            RETURN count(*) AS c
            """
        ).single()["c"]
        object_entities = session.run(
            """
            MATCH (f:Fact), (e:Entity {norm: f.object_norm})
            MERGE (f)-[:OBJECT_REFERS_TO]->(e)
            RETURN count(*) AS c
            """
        ).single()["c"]
        object_values = session.run(
            """
            MATCH (f:Fact), (v:Value {norm: f.object_norm})
            MERGE (f)-[:OBJECT_REFERS_TO]->(v)
            RETURN count(*) AS c
            """
        ).single()["c"]
    return {
        "subject_links": subject_entities + subject_values,
        "object_links": object_entities + object_values,
    }
```

- [ ] **Step 4: Run fact tests**

Run:

```bash
cd backend
PYTHONPATH=. uv run pytest tests/test_facts.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/ingest/facts.py backend/tests/test_facts.py
git commit -m "feat: link facts to graph anchors"
```

## Task 7: Fact Embeddings

**Files:**
- Modify: `backend/semantic_layer/ingest/embeddings.py`
- Modify: `backend/tests/test_embeddings.py`

- [ ] **Step 1: Write failing fake-embedding test**

Append to `backend/tests/test_embeddings.py`:

```python
@pytest.mark.neo4j
def test_embed_facts_sets_fake_vectors(neo4j_driver, monkeypatch):
    from semantic_layer.ingest.embeddings import embed_facts

    reset_graph(neo4j_driver)
    monkeypatch.setattr(settings, "fake_embeddings", True)
    with neo4j_driver.session(database=settings.neo4j_database) as session:
        session.run("CREATE (:Fact {id:'f1', text:'Blackwell / drove / growth'})")

    embed_facts(neo4j_driver)

    with neo4j_driver.session(database=settings.neo4j_database) as session:
        dim = session.run("MATCH (f:Fact {id:'f1'}) RETURN size(f.embedding) AS d").single()["d"]
    assert dim == settings.embedding_dimensions
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd backend
PYTHONPATH=. uv run pytest tests/test_embeddings.py::test_embed_facts_sets_fake_vectors -q
```

Expected: FAIL because `embed_facts` does not exist.

- [ ] **Step 3: Implement `embed_facts`**

Add to `backend/semantic_layer/ingest/embeddings.py`:

```python
def embed_facts(driver: Driver, batch: int = 64) -> None:
    """Embed Fact.text into Fact.embedding and ensure a vector index exists."""
    with driver.session(database=settings.neo4j_database) as session:
        rows = session.run(
            "MATCH (f:Fact) WHERE f.embedding IS NULL RETURN f.id AS id, f.text AS text"
        ).data()
        if settings.fake_embeddings:
            for i in range(0, len(rows), batch):
                window = rows[i:i + batch]
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (f:Fact {id: row.id})
                    CALL db.create.setNodeVectorProperty(f, 'embedding', row.vec)
                    """,
                    rows=[{"id": w["id"], "vec": fake_vector(w["text"] or "", settings.embedding_dimensions)}
                          for w in window],
                )
            _ensure_fact_vector_index(driver)
            return
    client = get_openai_client()
    with driver.session(database=settings.neo4j_database) as session:
        for i in range(0, len(rows), batch):
            window = rows[i:i + batch]
            vectors = client.embeddings.create(
                model=settings.embedding_model,
                input=[r["text"] for r in window],
                dimensions=settings.embedding_dimensions,
            ).data
            session.run(
                """
                UNWIND $rows AS row
                MATCH (f:Fact {id: row.id})
                CALL db.create.setNodeVectorProperty(f, 'embedding', row.vec)
                """,
                rows=[{"id": w["id"], "vec": v.embedding} for w, v in zip(window, vectors)],
            )
    _ensure_fact_vector_index(driver)


def _ensure_fact_vector_index(driver: Driver) -> None:
    with driver.session(database=settings.neo4j_database) as session:
        session.run(
            f"""
            CREATE VECTOR INDEX fact_embeddings IF NOT EXISTS
            FOR (f:Fact) ON (f.embedding)
            OPTIONS {{indexConfig: {{
              `vector.dimensions`: {settings.embedding_dimensions},
              `vector.similarity_function`: 'cosine'
            }}}}
            """
        )
```

- [ ] **Step 4: Run embedding tests**

Run:

```bash
cd backend
PYTHONPATH=. uv run pytest tests/test_embeddings.py -q
```

Expected: PASS, with OpenAI-marked tests skipped if no key is configured.

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/ingest/embeddings.py backend/tests/test_embeddings.py
git commit -m "feat: embed fact triplets"
```

## Task 8: Pipeline Fact Extraction

**Files:**
- Modify: `backend/semantic_layer/ingest/pipeline.py`
- Modify: `backend/tests/test_pipeline_entities.py`

- [ ] **Step 1: Write failing batch fact extraction test**

Append to `backend/tests/test_pipeline_entities.py`:

```python
def test_extract_facts_for_chunks_covers_all_rows(monkeypatch):
    rows = [{"id": f"c{i}", "text": f"text {i}"} for i in range(12)]
    seen_batches = []

    def fake_batch(texts):
        seen_batches.append(len(texts))
        return [[{"subject": t, "predicate": "mentions", "object": "NVIDIA", "text": f"{t} / mentions / NVIDIA",
                  "confidence": 1.0, "valid_from": None, "valid_until": None}] for t in texts]

    monkeypatch.setattr(pipe.settings, "entity_batch_size", 5)
    monkeypatch.setattr(pipe.settings, "ingest_max_workers", 3)
    monkeypatch.setattr(pipe, "extract_facts_batch", fake_batch)

    result = pipe.extract_facts_for_chunks(rows)
    assert set(result) == {f"c{i}" for i in range(12)}
    assert result["c7"][0]["subject"] == "text 7"
    assert sorted(seen_batches) == [2, 5, 5]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd backend
PYTHONPATH=. uv run pytest tests/test_pipeline_entities.py::test_extract_facts_for_chunks_covers_all_rows -q
```

Expected: FAIL because `extract_facts_for_chunks` does not exist.

- [ ] **Step 3: Wire fact extraction into pipeline**

Modify imports in `backend/semantic_layer/ingest/pipeline.py`:

```python
from semantic_layer.ingest.facts import extract_facts_batch
```

Add function mirroring `extract_entities_for_chunks`:

```python
def extract_facts_for_chunks(chunk_rows: list[dict]) -> dict[str, list[dict]]:
    size = max(1, settings.entity_batch_size)
    batches = [chunk_rows[i:i + size] for i in range(0, len(chunk_rows), size)]
    if not batches:
        return {}

    def run(batch: list[dict]) -> dict[str, list[dict]]:
        groups = extract_facts_batch([r["text"] for r in batch])
        return {r["id"]: facts for r, facts in zip(batch, groups)}

    out: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=settings.ingest_max_workers) as pool:
        for partial in pool.map(run, batches):
            out.update(partial)
    return out
```

Modify `_run_llm_stages` imports:

```python
    from semantic_layer.ingest.facts import load_facts, link_facts
    from semantic_layer.ingest.embeddings import embed_chunks, embed_tables, embed_facts
```

After `bridge_entities_to_values(driver)`:

```python
    facts_by_chunk = extract_facts_for_chunks(chunk_rows)
    for chunk_id, facts in facts_by_chunk.items():
        load_facts(driver, chunk_id, facts)
    link_facts(driver)
```

After `embed_tables(driver)`:

```python
    embed_facts(driver)
```

- [ ] **Step 4: Run pipeline entity tests**

Run:

```bash
cd backend
PYTHONPATH=. uv run pytest tests/test_pipeline_entities.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/ingest/pipeline.py backend/tests/test_pipeline_entities.py
git commit -m "feat: extract facts during ingest"
```

## Task 9: Search Facts Tool and Neighbor Facts

**Files:**
- Modify: `backend/semantic_layer/agent/graph_tools.py`
- Modify: `backend/tests/test_agent_graph_tools.py`

- [ ] **Step 1: Write failing graph tool tests**

Modify import in `backend/tests/test_agent_graph_tools.py`:

```python
    get_table_schema, search_facts,
```

Append:

```python
@pytest.mark.neo4j
def test_search_facts_returns_grounded_triplets(neo4j_driver):
    from semantic_layer.config import settings
    from semantic_layer.graph.client import reset_graph
    from semantic_layer.ingest.embeddings import fake_vector

    reset_graph(neo4j_driver)
    with neo4j_driver.session(database=settings.neo4j_database) as session:
        session.run(
            """
            CREATE (:Document {id:'doc:pr', title:'PR'})
            CREATE (:Chunk {id:'c1', doc_id:'doc:pr', ordinal:0, text:'Blackwell drove Data Center growth.'})
            CREATE (:Fact {
                id:'f1',
                subject:'Blackwell',
                predicate:'drove',
                object:'Data Center growth',
                text:'Blackwell / drove / Data Center growth',
                confidence:0.9,
                source_chunk_id:'c1'
            })
            WITH 1 AS _
            MATCH (c:Chunk {id:'c1'}), (f:Fact {id:'f1'})
            CREATE (c)-[:HAS_FACT]->(f)
            """
        )
        session.run(
            """
            MATCH (f:Fact {id:'f1'})
            CALL db.create.setNodeVectorProperty(f, 'embedding', $vec)
            """,
            vec=fake_vector("Blackwell Data Center growth", settings.embedding_dimensions),
        )

    data = json.loads(search_facts.invoke({"query": "Data Center growth", "limit": 5}))
    assert data[0]["subject"] == "Blackwell"
    assert data[0]["predicate"] == "drove"
    assert data[0]["chunk_id"] == "c1"


@pytest.mark.neo4j
def test_neighbors_includes_related_facts(neo4j_driver):
    from semantic_layer.config import settings
    from semantic_layer.graph.client import reset_graph

    reset_graph(neo4j_driver)
    with neo4j_driver.session(database=settings.neo4j_database) as session:
        session.run(
            """
            CREATE (:Fact {
              id:'f1',
              subject:'Blackwell',
              subject_norm:'blackwell',
              predicate:'drove',
              object:'Data Center growth',
              object_norm:'data center growth',
              text:'Blackwell / drove / Data Center growth',
              confidence:0.9,
              source_chunk_id:'c1'
            })
            """
        )
    data = json.loads(neighbors.invoke({"name": "Blackwell"}))
    assert data["facts"][0]["text"] == "Blackwell / drove / Data Center growth"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd backend
PYTHONPATH=. uv run pytest tests/test_agent_graph_tools.py::test_neighbors_includes_related_facts -q
```

Expected: FAIL because `neighbors` has no `facts` field.

- [ ] **Step 3: Add `search_facts` and update `neighbors`**

Modify `backend/semantic_layer/agent/graph_tools.py`.

Add import:

```python
from semantic_layer.ingest.embeddings import embed_query
```

Add tool:

```python
@tool
def search_facts(query: str, limit: int = 10) -> str:
    """Search atomic Fact triplets by vector similarity and return source provenance."""
    try:
        vector = embed_query(query)
        records = driver().execute_query(
            """
            CALL db.index.vector.queryNodes('fact_embeddings', $limit, $vector)
            YIELD node, score
            OPTIONAL MATCH (c:Chunk {id: node.source_chunk_id})
            RETURN node.id AS id, node.subject AS subject, node.predicate AS predicate,
                   node.object AS object, node.text AS text, node.confidence AS confidence,
                   node.source_chunk_id AS chunk_id, c.doc_id AS doc_id, c.ordinal AS ordinal,
                   score
            ORDER BY score DESC
            """,
            limit=limit,
            vector=vector,
            database_=settings.neo4j_database,
        ).records
    except Exception:
        records = driver().execute_query(
            """
            MATCH (f:Fact)
            WHERE toLower(f.text) CONTAINS toLower($query)
            OPTIONAL MATCH (c:Chunk {id: f.source_chunk_id})
            RETURN f.id AS id, f.subject AS subject, f.predicate AS predicate,
                   f.object AS object, f.text AS text, f.confidence AS confidence,
                   f.source_chunk_id AS chunk_id, c.doc_id AS doc_id, c.ordinal AS ordinal,
                   1.0 AS score
            ORDER BY f.confidence DESC
            LIMIT $limit
            """,
            query=query,
            limit=limit,
            database_=settings.neo4j_database,
        ).records
    return json.dumps([dict(r) for r in records])
```

In `neighbors`, add:

```python
    facts = driver().execute_query(
        """
        MATCH (f:Fact)
        WHERE f.subject_norm = $key OR f.object_norm = $key OR toLower(f.text) CONTAINS $key
        RETURN f.id AS id, f.text AS text, f.subject AS subject, f.predicate AS predicate,
               f.object AS object, f.confidence AS confidence, f.source_chunk_id AS chunk_id
        ORDER BY f.confidence DESC LIMIT 10
        """,
        key=key, database_=settings.neo4j_database,
    ).records
```

Return:

```python
        "facts": [dict(r) for r in facts],
```

- [ ] **Step 4: Run graph tool tests**

Run:

```bash
cd backend
PYTHONPATH=. uv run pytest tests/test_agent_graph_tools.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/agent/graph_tools.py backend/tests/test_agent_graph_tools.py
git commit -m "feat: expose fact search tools"
```

## Task 10: Expose Subtype in Web Graph API

**Files:**
- Modify: `backend/semantic_layer/web/graph_api.py`
- Modify: `backend/tests/test_web_graph_api.py`

- [ ] **Step 1: Write failing graph projection test**

Append to `backend/tests/test_web_graph_api.py`:

```python
@pytest.mark.neo4j
def test_schema_graph_includes_entity_subtype(neo4j_driver):
    from semantic_layer.config import settings
    from semantic_layer.graph.client import reset_graph
    from semantic_layer.web.graph_api import get_schema_graph

    reset_graph(neo4j_driver)
    with neo4j_driver.session(database=settings.neo4j_database) as session:
        session.run(
            """
            CREATE (:Document {id:'doc:pr', title:'PR'})
            CREATE (:Chunk {id:'c1', doc_id:'doc:pr', ordinal:0, text:'Blackwell drove growth.'})
            CREATE (:Entity {norm:'blackwell', name:'Blackwell', label:'Object'})
            CREATE (:OntologySubtype {name:'ProductArchitecture', base_type:'Object'})
            WITH 1 AS _
            MATCH (d:Document {id:'doc:pr'}), (c:Chunk {id:'c1'}),
                  (e:Entity {norm:'blackwell'}), (s:OntologySubtype {name:'ProductArchitecture'})
            CREATE (d)-[:HAS_CHUNK]->(c)
            CREATE (c)-[:MENTIONS]->(e)
            CREATE (e)-[:INSTANCE_OF]->(s)
            """
        )
    graph = get_schema_graph(source="documents", max_chunks=10)
    entity = next(n for n in graph["nodes"] if n["id"] == "entity:blackwell")
    assert entity["entityType"] == "Object"
    assert entity["subtype"] == "ProductArchitecture"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd backend
PYTHONPATH=. uv run pytest tests/test_web_graph_api.py::test_schema_graph_includes_entity_subtype -q
```

Expected: FAIL because entity nodes do not include `subtype`.

- [ ] **Step 3: Update entity projection query**

Modify `backend/semantic_layer/web/graph_api.py` entity query:

```cypher
        MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
        WHERE c.id IN $chunk_ids
        OPTIONAL MATCH (e)-[:INSTANCE_OF]->(s:OntologySubtype)
        WITH e, s, collect(DISTINCT c.id) AS chunk_ids
        WHERE size(chunk_ids) >= 2 OR exists((e)-[:REFERS_TO]->(:Value)) OR s IS NOT NULL
        UNWIND chunk_ids AS chunk_id
        RETURN chunk_id, e.norm AS norm, e.name AS name, e.label AS label,
               s.name AS subtype
```

Update node creation:

```python
        nodes.setdefault(eid, {"id": eid, "label": r["name"], "kind": "entity",
                               "source": "documents", "entityType": r["label"],
                               "subtype": r["subtype"]})
```

- [ ] **Step 4: Run web graph tests**

Run:

```bash
cd backend
PYTHONPATH=. uv run pytest tests/test_web_graph_api.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/web/graph_api.py backend/tests/test_web_graph_api.py
git commit -m "feat: expose ontology subtypes in graph api"
```

## Task 11: Final Verification

**Files:**
- No new files.

- [ ] **Step 1: Run focused backend tests**

Run:

```bash
cd backend
PYTHONPATH=. uv run pytest tests/test_ontology.py tests/test_entities_batch.py tests/test_doc_graph_bridge.py tests/test_facts.py tests/test_pipeline_entities.py tests/test_agent_graph_tools.py tests/test_web_graph_api.py -q
```

Expected: PASS, with environment-gated tests skipped when Neo4j/OpenAI are unavailable.

- [ ] **Step 2: Run broader backend suite**

Run:

```bash
cd backend
PYTHONPATH=. uv run pytest -q
```

Expected: PASS or only documented skips for unavailable external services.

- [ ] **Step 3: Inspect git status**

Run:

```bash
git status --short
```

Expected: no unstaged implementation files. If verification created cache files, remove only generated cache artifacts.

- [ ] **Step 4: Commit final test fixes if needed**

If Step 1 or Step 2 required small fixes after the last feature commit:

```bash
git add backend/semantic_layer backend/tests
git commit -m "test: stabilize ontology context graph"
```

If no fixes were needed, do not create an empty commit.
