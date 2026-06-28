# Ontology-Aware Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing graph retrieval tools use POLE+O ontology subtype edges for catalog routing, entity neighborhood context, and fact result enrichment.

**Architecture:** Keep the implementation inside `semantic_layer.agent.graph_tools`, where search/routing tools already live. Add ontology traversal Cypher to the existing tools, preserving current response shapes and adding optional subtype metadata where available.

**Tech Stack:** Python, LangChain tools, Neo4j Cypher, pytest.

---

### Task 1: Catalog Search Ontology Hits

**Files:**
- Modify: `backend/semantic_layer/agent/graph_tools.py`
- Test: `backend/tests/test_agent_graph_tools.py`

- [ ] **Step 1: Write the failing test**

Add this test to `backend/tests/test_agent_graph_tools.py`:

```python
def test_search_catalog_uses_ontology_subtypes_to_route_values(neo4j_driver):
    from semantic_layer.graph.client import reset_graph

    reset_graph(neo4j_driver)
    with neo4j_driver.session() as session:
        session.run(
            """
            CREATE (t:Table {id:'table:sales_pg.sales.architecture', name:'architecture'})
            CREATE (c:Column {id:'column:sales_pg.sales.architecture.name', name:'name'})
            CREATE (v:Value {name:'Blackwell', norm:'blackwell'})
            CREATE (e:Entity {name:'Blackwell', norm:'blackwell', label:'Object'})
            CREATE (s:OntologySubtype {
              name:'ProductArchitecture',
              base_type:'Object',
              domain:'product',
              description:'GPU product architecture'
            })
            CREATE (t)-[:HAS_COLUMN]->(c)
            CREATE (c)-[:HAS_VALUE]->(v)
            CREATE (e)-[:REFERS_TO]->(v)
            CREATE (e)-[:INSTANCE_OF]->(s)
            """
        )

    data = json.loads(search_catalog.invoke({"query": "product architecture"}))
    hit = next(row for row in data if row["kind"] == "ontology")

    assert hit["table_id"] == "table:sales_pg.sales.architecture"
    assert hit["column"] == "name"
    assert hit["name"] == "Blackwell"
    assert hit["subtype"] == "ProductArchitecture"
    assert hit["base_type"] == "Object"
```

- [ ] **Step 2: Run test to verify it fails**

Run from `backend`:

```bash
PYTHONPATH=. uv run pytest tests/test_agent_graph_tools.py::test_search_catalog_uses_ontology_subtypes_to_route_values -q
```

Expected: fail because no `kind: "ontology"` hit is returned.

- [ ] **Step 3: Implement minimal catalog ontology hits**

In `search_catalog`, add a Cypher query that matches query terms against `OntologySubtype.name`, `OntologySubtype.domain`, `OntologySubtype.description`, and linked `Entity.name`, then traverses through `Entity -> Value -> Column -> Table`. Merge these records into the existing `hits` list before truncating.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONPATH=. uv run pytest tests/test_agent_graph_tools.py::test_search_catalog_uses_ontology_subtypes_to_route_values -q
```

Expected: pass.

### Task 2: Neighbor Ontology Context

**Files:**
- Modify: `backend/semantic_layer/agent/graph_tools.py`
- Test: `backend/tests/test_agent_graph_tools.py`

- [ ] **Step 1: Write the failing test**

Add this test to `backend/tests/test_agent_graph_tools.py`:

```python
def test_neighbors_includes_entity_ontology_context(neo4j_driver):
    from semantic_layer.graph.client import reset_graph

    reset_graph(neo4j_driver)
    with neo4j_driver.session() as session:
        session.run(
            """
            CREATE (ch:Chunk {id:'chunk:1', doc_id:'doc:nvidia', ordinal:0})
            CREATE (e:Entity {name:'Blackwell', norm:'blackwell', label:'Object'})
            CREATE (s:OntologySubtype {
              name:'ProductArchitecture',
              base_type:'Object',
              domain:'product',
              description:'GPU product architecture'
            })
            CREATE (ch)-[:MENTIONS]->(e)
            CREATE (e)-[:INSTANCE_OF]->(s)
            """
        )

    data = json.loads(neighbors.invoke({"name": "Blackwell"}))

    assert data["documents"][0]["entityType"] == "Object"
    assert data["documents"][0]["subtype"] == "ProductArchitecture"
    assert data["documents"][0]["subtypeDescription"] == "GPU product architecture"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=. uv run pytest tests/test_agent_graph_tools.py::test_neighbors_includes_entity_ontology_context -q
```

Expected: fail because document rows currently only include `doc_id` and `chunks`.

- [ ] **Step 3: Implement neighbor enrichment**

Update the `neighbors` document query to optionally match `Entity-[:INSTANCE_OF]->OntologySubtype`, collect subtype metadata, and return deterministic `entityType`, `subtype`, and `subtypeDescription` fields.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONPATH=. uv run pytest tests/test_agent_graph_tools.py::test_neighbors_includes_entity_ontology_context -q
```

Expected: pass.

### Task 3: Fact Search Ontology Context

**Files:**
- Modify: `backend/semantic_layer/agent/graph_tools.py`
- Test: `backend/tests/test_agent_graph_tools.py`

- [ ] **Step 1: Write the failing test**

Add this test to `backend/tests/test_agent_graph_tools.py`:

```python
def test_search_facts_includes_subject_object_ontology_context(neo4j_driver, monkeypatch):
    from semantic_layer.graph.client import reset_graph
    from semantic_layer.agent import graph_tools

    reset_graph(neo4j_driver)
    monkeypatch.setattr(graph_tools, "embed_query", lambda _query: [0.1, 0.2, 0.3])
    with neo4j_driver.session() as session:
        session.run(
            """
            CREATE (ch:Chunk {id:'chunk:1', doc_id:'doc:nvidia', ordinal:0})
            CREATE (f:Fact {
              id:'fact:1',
              subject:'Blackwell',
              subject_norm:'blackwell',
              predicate:'drove',
              object:'growth',
              object_norm:'growth',
              text:'Blackwell drove growth',
              confidence:0.91,
              source_chunk_id:'chunk:1'
            })
            CREATE (subject:Entity {name:'Blackwell', norm:'blackwell', label:'Object'})
            CREATE (subjectSubtype:OntologySubtype {
              name:'ProductArchitecture',
              base_type:'Object',
              domain:'product',
              description:'GPU product architecture'
            })
            CREATE (objectEntity:Entity {name:'growth', norm:'growth', label:'Event'})
            CREATE (objectSubtype:OntologySubtype {
              name:'FinancialResult',
              base_type:'Event',
              domain:'finance',
              description:'reported financial outcome'
            })
            CREATE (ch)-[:HAS_FACT]->(f)
            CREATE (ch)-[:MENTIONS]->(subject)
            CREATE (ch)-[:MENTIONS]->(objectEntity)
            CREATE (subject)-[:INSTANCE_OF]->(subjectSubtype)
            CREATE (objectEntity)-[:INSTANCE_OF]->(objectSubtype)
            """
        )

    data = json.loads(search_facts.invoke({"query": "Blackwell growth"}))
    row = data[0]

    assert row["subject_entity_type"] == "Object"
    assert row["subject_subtype"] == "ProductArchitecture"
    assert row["object_entity_type"] == "Event"
    assert row["object_subtype"] == "FinancialResult"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=. uv run pytest tests/test_agent_graph_tools.py::test_search_facts_includes_subject_object_ontology_context -q
```

Expected: fail because fact search results do not include ontology fields.

- [ ] **Step 3: Implement fact enrichment**

Update both vector and text-fallback `search_facts` queries to optionally match subject and object entities by normalized fact endpoints, then return `subject_entity_type`, `subject_subtype`, `object_entity_type`, and `object_subtype`.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONPATH=. uv run pytest tests/test_agent_graph_tools.py::test_search_facts_includes_subject_object_ontology_context -q
```

Expected: pass.

### Task 4: Regression Pass and Commit

**Files:**
- Modify: `backend/semantic_layer/agent/graph_tools.py`
- Test: `backend/tests/test_agent_graph_tools.py`

- [ ] **Step 1: Run focused graph tool tests**

Run:

```bash
PYTHONPATH=. uv run pytest tests/test_agent_graph_tools.py -q
```

Expected: all tests in the file pass, or unrelated environment failures are recorded with exact error text.

- [ ] **Step 2: Run formatting/diff checks**

Run:

```bash
git diff --check -- backend/semantic_layer/agent/graph_tools.py backend/tests/test_agent_graph_tools.py
git status --short
```

Expected: no whitespace errors; only intended files changed.

- [ ] **Step 3: Commit**

Run:

```bash
git add backend/semantic_layer/agent/graph_tools.py backend/tests/test_agent_graph_tools.py docs/superpowers/plans/2026-06-28-ontology-aware-retrieval.md
git commit -m "feat: use ontology in graph retrieval"
```

Expected: commit succeeds.
