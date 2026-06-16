"""Load a SchemaBundle into Neo4j via NeoCarta's Neo4jRDBMSLoader (idempotent)."""

from neo4j import Driver

from neocarta.ingest.rdbms import Neo4jRDBMSLoader

from semantic_layer.config import settings
from semantic_layer.ingest.sql_extractor import SchemaBundle


def load_bundle(driver: Driver, bundle: SchemaBundle) -> None:
    loader = Neo4jRDBMSLoader(driver, database_name=settings.neo4j_database)
    if bundle.databases:
        loader.load_database_nodes(bundle.databases, overwrite_existing=True)
    if bundle.schemas:
        loader.load_schema_nodes(bundle.schemas, overwrite_existing=True)
    if bundle.tables:
        loader.load_table_nodes(bundle.tables, overwrite_existing=True)
    if bundle.columns:
        loader.load_column_nodes(bundle.columns, overwrite_existing=True)
    if bundle.has_schema:
        loader.load_has_schema_relationships(bundle.has_schema)
    if bundle.has_table:
        loader.load_has_table_relationships(bundle.has_table)
    if bundle.has_column:
        loader.load_has_column_relationships(bundle.has_column)
    if bundle.references:
        loader.load_references_relationships(bundle.references)
