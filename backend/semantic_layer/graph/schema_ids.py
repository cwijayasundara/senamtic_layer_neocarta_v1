"""Deterministic, collision-free id helpers for graph nodes.

Ids are stable strings so re-ingestion MERGEs onto the same nodes.
"""


def database_id(source: str) -> str:
    return f"db:{source}"


def schema_id(source: str, schema: str) -> str:
    return f"schema:{source}.{schema}"


def table_id(source: str, schema: str, table: str) -> str:
    return f"table:{source}.{schema}.{table}"


def column_id(source: str, schema: str, table: str, column: str) -> str:
    return f"col:{source}.{schema}.{table}.{column}"
