# Ontology-Aware Retrieval Design

## Goal

Use the existing POLE+O ontology as an active retrieval signal in the context graph and metadata graph. Ontology should help route user questions to relevant catalog tables, document entities, and extracted facts, not only annotate graph visualization payloads.

## Scope

This follow-up keeps the change inside the existing agent graph tools:

- `search_catalog` returns ontology-derived hits when query terms match ontology subtype names, domains, descriptions, or related entity names.
- `neighbors` includes ontology context for matched document entities.
- `search_facts` enriches fact results with ontology context for subject/object anchors when available.

The change does not alter ingestion prompts, planner architecture, SQL generation, graph schema shape, or frontend layout.

## Current State

Ingestion loads `OntologySubtype` nodes and links extracted `Entity` nodes with `INSTANCE_OF`. The graph API exposes subtype information for visualization. The search tools currently use catalog names, business terms, values, vector fact search, and direct entity/value matches. They do not search ontology nodes or traverse subtype edges as part of retrieval.

## Proposed Behavior

### Catalog Search

When a user searches for an ontology concept such as `product architecture`, `fiscal period`, or `support incident`, `search_catalog` should return hits derived from:

```text
OntologySubtype <-[:INSTANCE_OF]- Entity -[:REFERS_TO]-> Value <-[:HAS_VALUE]- Column <-[:HAS_COLUMN]- Table
```

Returned hits should use `kind: "ontology"` and include:

- `id`: ontology subtype name
- `name`: entity or value name that connected the subtype to the catalog
- `table_id`
- `column`, when a value bridge exists
- `subtype`
- `base_type`
- `score`

These hits should be merged with existing column, table, value, and business-term hits. Existing hit types keep their current shape.

### Neighbors

`neighbors(name)` should return ontology metadata for matching document entities:

- `entityType`
- `subtype`
- `subtypeDescription`

This gives the agent and UI enough context to distinguish, for example, `Blackwell` as an `Object/ProductArchitecture`, not just a string that appears in a document and catalog value.

### Fact Search

`search_facts` should add best-effort ontology context to fact results when the fact subject or object resolves to a typed entity:

- `subject_entity_type`
- `subject_subtype`
- `object_entity_type`
- `object_subtype`

The initial implementation should enrich results without changing vector ranking. Ranking changes can follow after we have retrieval examples that need boosting.

## Error Handling

Ontology traversal is optional. If a fact or entity has no subtype, existing behavior should remain unchanged and returned subtype fields should be `null` or omitted only where the existing response shape already omits optional data.

Search should continue to work when ontology nodes are absent, which is useful for partial test fixtures and older local graph data.

## Testing

Add tests that prove:

- `search_catalog("product architecture")` can route through ontology subtype edges to the table containing a bridged value.
- `neighbors("Blackwell")` includes entity type and subtype metadata.
- `search_facts(...)` returns subject/object subtype context for typed fact anchors.

Tests should use focused Neo4j fixtures and should avoid requiring external LLM or embedding calls where possible.
