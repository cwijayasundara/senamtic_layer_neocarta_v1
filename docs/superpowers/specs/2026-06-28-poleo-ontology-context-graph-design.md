# POLE+O Ontology Context Graph — Design

**Date:** 2026-06-28
**Status:** Draft
**Builds on:** [Value Layer + Document Context Graph](2026-06-16-value-layer-and-doc-context-graph-design.md),
[Graph-Native Planner + Parallel Legs](2026-06-17-graph-native-planner-design.md)

## Problem

The current document graph already extracts POLE+O entities from chunks and bridges them
to structured catalog values:

```text
Document -> Chunk -> Entity -> Value -> Column -> Table
```

That is enough to connect documents to SQL/API metadata, but the ontology is still too
flat. `Entity.label` is one of `Person`, `Org`, `Location`, `Event`, or `Object`; it does
not preserve the domain-specific type the planner actually needs. In the NVIDIA demo,
important concepts such as `Blackwell`, `Data Center`, `DGX Cloud`, `Cloud Service
Provider`, `EMEA`, fiscal quarters, support tickets, and revenue metrics all collapse into
generic labels. The graph can say they are things, but not what kind of things they are.

This limits three workflows:

- **Planning:** the planner cannot distinguish a product architecture from a business
  segment or a metric without re-inferring that meaning from table/value context.
- **Retrieval:** facts stated in documents remain embedded in chunks. Atomic claims such
  as "Blackwell drove Data Center growth" are not independently searchable or reusable.
- **Ontology evolution:** extraction errors and generic `Object` labels are not captured
  as signals for improving the schema.

## Approach

Adopt the article's posture: keep POLE+O as a small fixed base ontology, then grow
domain-specific subtypes only when the data shows they are useful. Do not replace the
existing graph. Add ontology metadata and a Fact fallback beside the current entity/value
layer.

The target graph becomes:

```text
Document -> Chunk -> Entity -> Value -> Column -> Table
                 |        \
                 |         -> Fact <- Chunk
                 v
           OntologySubtype -> OntologyType
```

## Graph Model

### Base ontology nodes

- `(:OntologyType {name})`
  - One of `Person`, `Org`, `Location`, `Event`, `Object`.
  - This is the fixed POLE+O base.

- `(:OntologySubtype {name, base_type, domain, description})`
  - A lightweight extension under one base type.
  - Loaded from a checked-in ontology catalog.

- `(:OntologySubtype)-[:SUBTYPE_OF]->(:OntologyType)`

- `(:Entity)-[:INSTANCE_OF]->(:OntologySubtype)`

`Entity.label` remains during migration and mirrors the base type, so existing tools and
UI code continue to work.

### Initial subtype catalog

Keep the catalog deliberately small:

| Base type | Initial subtypes |
|---|---|
| `Object` | `Product`, `ProductArchitecture`, `Technology`, `Metric`, `DocumentArtifact` |
| `Org` | `Customer`, `Partner`, `BusinessUnit`, `Vendor` |
| `Location` | `Region`, `Country` |
| `Event` | `FiscalPeriod`, `PressRelease`, `SupportIncident`, `SalesTransaction` |
| `Person` | none initially |

This gives the planner useful type information without trying to model the whole
enterprise domain.

### Fact nodes

Add `:Fact` as the fallback primitive for useful claims that do not belong as typed graph
edges yet.

Properties:

- `id`: stable hash of source chunk, subject, predicate, object
- `subject`
- `predicate`
- `object`
- `text`: concatenated triplet for search and embedding
- `confidence`
- `source_chunk_id`
- `valid_from`
- `valid_until`

Relationships:

- `(:Chunk)-[:HAS_FACT]->(:Fact)`
- `(:Fact)-[:SUBJECT_REFERS_TO]->(:Entity|Value)` when resolvable
- `(:Fact)-[:OBJECT_REFERS_TO]->(:Entity|Value)` when resolvable

Facts are primarily retrieved through vector and text search. They may be bridged to
entities/values when there is a clear match, but they do not need to be fully wired into
the ontology before they are useful.

## Components

### `ingest/ontology.py`

New module responsible for:

- loading the checked-in subtype catalog;
- MERGEing `OntologyType` and `OntologySubtype` nodes;
- linking subtypes to base types;
- linking extracted entities to subtypes;
- validating that every subtype maps to one POLE+O base type.

### Ontology catalog

Add `backend/semantic_layer/ingest/ontology_catalog.json`.

The catalog is static, reviewable, and intentionally small. The extraction prompt can use
it as the only allowed subtype vocabulary. New subtypes are added through review, not
invented silently during production ingest.

### `ingest/entities.py`

Extend extraction output from:

```json
{"name": "Blackwell", "label": "Object"}
```

to:

```json
{
  "name": "Blackwell",
  "base_type": "Object",
  "subtype": "ProductArchitecture",
  "confidence": 0.92,
  "evidence": "..."
}
```

Compatibility rules:

- If `base_type` is missing but `label` exists, use `label`.
- If subtype is missing, invalid, or has `confidence < 0.80`, keep the entity at the base
  POLE+O type with no `INSTANCE_OF` subtype edge.
- Preserve existing deduplication by normalized entity name.

### `ingest/facts.py`

New module responsible for:

- extracting atomic fact triplets from chunks;
- cleaning and validating LLM output;
- creating stable Fact ids;
- loading `Chunk -> Fact` edges;
- linking fact subjects/objects to existing `Entity` or `Value` nodes by normalized name;
- embedding facts for vector search.

### `agent/graph_tools.py`

Add a `search_facts(query, limit=10)` graph tool. It returns concise, grounded triplets
with source document/chunk provenance.

Update `neighbors(name)` to include facts where the name appears as subject or object.

### `agent/planner.py`

Use subtype metadata as an additional planning signal:

- `ProductArchitecture` helps route `Blackwell` to architecture/product joins.
- `Metric` helps detect revenue, ARR, ticket count, and utilization questions.
- `FiscalPeriod` helps combine document periods with structured sales data.
- `SupportIncident` helps route ITSM-style questions.

Planner behavior remains deterministic where possible; subtypes reduce LLM guesswork
without adding a new ReAct loop.

### `web/graph_api.py`

Expose subtype metadata in node payloads:

- `entityType`: POLE+O base type
- `subtype`: ontology subtype, if present

No UI redesign is required for the first implementation. Existing graph rendering receives
the extra fields without changing layout or node filtering.

## Data Flow

```text
run_ingest:
  reset
  -> SQL/API metadata
  -> values and periods
  -> bridge sources
  -> documents and chunks
  -> load ontology catalog
  -> extract typed entities
  -> load entities and INSTANCE_OF subtype edges
  -> bridge entities to values
  -> extract facts
  -> link facts to entities/values
  -> embed chunks, tables, and facts
```

`with_llm=False` skips typed entity extraction and fact extraction, but the ontology
catalog can still be loaded because it is deterministic.

## Ontology Discovery Loop

After ingest, add a diagnostic report rather than production auto-mutation:

- generic `Object` entities with high mention counts;
- entities with invalid or missing subtype;
- subtype assignments with `confidence < 0.80`;
- frequent Fact predicates not represented by graph relationships;
- clusters of values/entities that share table context but lack subtype coverage.

The report proposes candidate subtypes, but does not write them into the catalog. The
catalog remains human-reviewed to keep the ontology small and stable.

## Error Handling

- Invalid LLM JSON returns empty extraction for that batch, matching the current entity
  extractor behavior.
- Invalid subtypes degrade to base POLE+O labels.
- Fact extraction failures do not block entity loading, embeddings, or table indexing.
- Fact subject/object linking is best-effort. Unlinked facts remain searchable.
- Re-ingest must be idempotent through stable ids and MERGE semantics.

## Testing

- Catalog loading creates all five `OntologyType` nodes and the initial subtype set.
- Invalid catalog entries fail fast in unit tests.
- Entity cleaning accepts legacy `{name,label}` and new typed entity objects.
- Entity loading creates `INSTANCE_OF` for valid subtypes and preserves `Entity.label`.
- Invalid subtype assignments and assignments with `confidence < 0.80` degrade to base
  labels.
- Fact extraction cleaning accepts valid triplets and rejects malformed rows.
- Fact loading is idempotent and creates `Chunk -> Fact`.
- Fact linking creates subject/object links when matching `Entity.norm` or `Value.norm`
  exists.
- `search_facts("Blackwell Data Center growth")` returns grounded triplets with chunk
  provenance in a seeded graph.
- `neighbors("Blackwell")` includes catalog hits, document mentions, and related facts.
- Pipeline ingest with `with_llm=False` still loads ontology catalog deterministically.
- Pipeline ingest with LLM stages loads typed entities and facts without breaking existing
  value bridge and planner tests.

## Phasing

1. **Ontology catalog + typed entities**
   - Add catalog, loader, typed extraction schema, subtype links, and tests.
   - Preserve current graph behavior.

2. **Fact fallback**
   - Add Fact extraction, loading, linking, embeddings, and `search_facts`.

3. **Planner and UI usage**
   - Teach planner to use subtype metadata.
   - Surface subtype/fact context in graph API and answer provenance.

## Out of Scope

- Auto-generating and auto-applying new ontology subtypes during production ingest.
- Replacing NeoCarta metadata nodes.
- Modeling every possible document claim as a typed relationship.
- Full GraphRAG community detection or summarization.
- UI redesign beyond exposing subtype/fact metadata.
