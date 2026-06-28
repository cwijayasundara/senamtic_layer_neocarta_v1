"""Introspect live SQL databases into NeoCarta metadata-layer model objects."""

import sqlite3
from contextlib import closing
from dataclasses import dataclass, field

import psycopg

from neocarta.data_model.schema.rdbms import (
    Database, Schema, Table, Column, HasSchema, HasTable, HasColumn, References,
)

from semantic_layer.graph.schema_ids import (
    database_id, schema_id, table_id, column_id,
)


@dataclass
class SchemaBundle:
    databases: list = field(default_factory=list)
    schemas: list = field(default_factory=list)
    tables: list = field(default_factory=list)
    columns: list = field(default_factory=list)
    has_schema: list = field(default_factory=list)
    has_table: list = field(default_factory=list)
    has_column: list = field(default_factory=list)
    references: list = field(default_factory=list)


def _bundle_for(source: str, schema_name: str, platform: str) -> SchemaBundle:
    b = SchemaBundle()
    b.databases.append(Database(id=database_id(source), name=source, platform=platform))
    b.schemas.append(Schema(id=schema_id(source, schema_name), name=schema_name))
    b.has_schema.append(HasSchema(database_id=database_id(source), schema_id=schema_id(source, schema_name)))
    return b


def extract_postgres(dsn: str, source: str = "sales_pg", schema_name: str = "sales") -> SchemaBundle:
    b = _bundle_for(source, schema_name, platform="postgresql")
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = %s ORDER BY table_name",
            (schema_name,),
        )
        tables = [r[0] for r in cur.fetchall()]

        cur.execute(
            "SELECT table_name, column_name, data_type, is_nullable "
            "FROM information_schema.columns WHERE table_schema = %s",
            (schema_name,),
        )
        col_rows = cur.fetchall()

        cur.execute(
            """
            SELECT tc.table_name, kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
            WHERE tc.table_schema = %s AND tc.constraint_type = 'PRIMARY KEY'
            """,
            (schema_name,),
        )
        pks = {(t, c) for t, c in cur.fetchall()}

        cur.execute(
            """
            SELECT kcu.table_name, kcu.column_name,
                   ccu.table_name AS ref_table, ccu.column_name AS ref_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
              ON tc.constraint_name = ccu.constraint_name AND tc.table_schema = ccu.table_schema
            WHERE tc.table_schema = %s AND tc.constraint_type = 'FOREIGN KEY'
            """,
            (schema_name,),
        )
        fks = cur.fetchall()
    fk_cols = {(t, c) for t, c, _, _ in fks}

    for t in tables:
        b.tables.append(Table(id=table_id(source, schema_name, t), name=t))
        b.has_table.append(HasTable(schema_id=schema_id(source, schema_name), table_id=table_id(source, schema_name, t)))
    for t, c, dtype, nullable in col_rows:
        cid = column_id(source, schema_name, t, c)
        b.columns.append(Column(
            id=cid, name=c, type=dtype, nullable=(nullable == "YES"),
            is_primary_key=(t, c) in pks, is_foreign_key=(t, c) in fk_cols,
        ))
        b.has_column.append(HasColumn(table_id=table_id(source, schema_name, t), column_id=cid))
    for t, c, rt, rc in fks:
        b.references.append(References(
            source_column_id=column_id(source, schema_name, t, c),
            target_column_id=column_id(source, schema_name, rt, rc),
            criteria=f"{t}.{c} -> {rt}.{rc}",
        ))
    return b


def extract_sqlite(db_path: str, source: str, schema_name: str = "main") -> SchemaBundle:
    b = _bundle_for(source, schema_name, platform="sqlite")
    # `closing` guarantees the connection is closed even if introspection raises.
    with closing(sqlite3.connect(db_path)) as con:
        con.row_factory = sqlite3.Row
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()]
        for t in tables:
            b.tables.append(Table(id=table_id(source, schema_name, t), name=t))
            b.has_table.append(HasTable(schema_id=schema_id(source, schema_name), table_id=table_id(source, schema_name, t)))
            # `t` comes from sqlite_master (not caller input); PRAGMA does not
            # support parameter binding, so interpolation is safe here.
            info = con.execute(f"PRAGMA table_info({t})").fetchall()
            fk_list = con.execute(f"PRAGMA foreign_key_list({t})").fetchall()
            fk_cols = {row["from"] for row in fk_list}
            for row in info:
                cid = column_id(source, schema_name, t, row["name"])
                b.columns.append(Column(
                    id=cid, name=row["name"], type=row["type"] or "TEXT",
                    nullable=(row["notnull"] == 0),
                    is_primary_key=bool(row["pk"]), is_foreign_key=row["name"] in fk_cols,
                ))
                b.has_column.append(HasColumn(table_id=table_id(source, schema_name, t), column_id=cid))
            for row in fk_list:
                b.references.append(References(
                    source_column_id=column_id(source, schema_name, t, row["from"]),
                    target_column_id=column_id(source, schema_name, row["table"], row["to"]),
                    criteria=f"{t}.{row['from']} -> {row['table']}.{row['to']}",
                ))
    return b
