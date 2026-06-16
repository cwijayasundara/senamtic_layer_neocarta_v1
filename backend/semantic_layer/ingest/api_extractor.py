"""Introspect mock-API OpenAPI specs into NeoCarta metadata models.

API  -> Database (platform='rest-api')
Endpoint (method+path) -> Table
Response-schema property -> Column
"""

from neocarta.data_model.rdbms import (
    Database, Schema, Table, Column, HasSchema, HasTable, HasColumn,
)

from semantic_layer.graph.schema_ids import (
    database_id, schema_id, table_id, column_id,
)
from semantic_layer.ingest.sql_extractor import SchemaBundle

_SCHEMA = "api"


def _resolve_item_schema(operation: dict, components: dict) -> dict | None:
    """Return the object schema of a 200 response (unwrapping array + $ref)."""
    try:
        content = operation["responses"]["200"]["content"]["application/json"]["schema"]
    except KeyError:
        return None
    if content.get("type") == "array":
        content = content.get("items", {})
    ref = content.get("$ref")
    if ref:
        name = ref.split("/")[-1]
        return components.get("schemas", {}).get(name)
    return content if content.get("properties") else None


def extract_openapi(spec: dict, source: str) -> SchemaBundle:
    b = SchemaBundle()
    b.databases.append(Database(id=database_id(source), name=source, platform="rest-api"))
    b.schemas.append(Schema(id=schema_id(source, _SCHEMA), name=_SCHEMA))
    b.has_schema.append(HasSchema(database_id=database_id(source), schema_id=schema_id(source, _SCHEMA)))

    components = spec.get("components", {})
    for path, methods in spec.get("paths", {}).items():
        for method, operation in methods.items():
            endpoint = f"{method.upper()} {path}"
            tid = table_id(source, _SCHEMA, endpoint)
            b.tables.append(Table(id=tid, name=endpoint, description=operation.get("summary")))
            b.has_table.append(HasTable(schema_id=schema_id(source, _SCHEMA), table_id=tid))
            item = _resolve_item_schema(operation, components)
            if not item:
                continue
            for prop, meta in item.get("properties", {}).items():
                cid = column_id(source, _SCHEMA, endpoint, prop)
                b.columns.append(Column(
                    id=cid, name=prop, type=meta.get("type", "string"),
                    nullable=True, is_primary_key=False,
                    is_foreign_key=prop.endswith("_id"),
                ))
                b.has_column.append(HasColumn(table_id=tid, column_id=cid))
    return b


def extract_all_apis(spec_getter, sources) -> list:
    """spec_getter(prefix) -> openapi dict. sources: iterable of api names."""
    return [extract_openapi(spec_getter(f"/{s}"), source=s) for s in sources]
