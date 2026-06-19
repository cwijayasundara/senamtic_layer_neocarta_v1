"""Turn synthetic ApiDef entries into OpenAPI 3.1 spec dicts that extract_openapi
can introspect, so distractor REST endpoints land in the graph on the real path."""

from data.generators.scale_catalog import ApiDef, ScaleCatalog
from semantic_layer.ingest.api_extractor import extract_openapi
from semantic_layer.ingest.sql_extractor import SchemaBundle


def build_openapi_spec(api: ApiDef) -> dict:
    model = api.resource.replace("_", " ").title().replace(" ", "")
    return {
        "openapi": "3.1.0",
        "info": {"title": api.source, "version": "1.0.0"},
        "paths": {
            f"/{api.resource}": {
                "get": {
                    "summary": f"List {api.resource}",
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {"$ref": f"#/components/schemas/{model}"},
                                    }
                                }
                            }
                        }
                    },
                }
            }
        },
        "components": {
            "schemas": {
                model: {
                    "type": "object",
                    "properties": {f.name: {"type": f.type} for f in api.fields},
                }
            }
        },
    }


def extract_synthetic_apis(catalog: ScaleCatalog) -> list[SchemaBundle]:
    return [extract_openapi(build_openapi_spec(a), source=a.source) for a in catalog.apis]
