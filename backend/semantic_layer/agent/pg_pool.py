"""Process-wide pooled Postgres connections for the SQL tool.

Replaces sql_tools' per-call psycopg.connect with a ConnectionPool so repeated
queries reuse warm connections. Constructed with open=False so importing this
module (and unit tests) never touches the database; the pool opens lazily on
first use in sql_tools._run."""

from functools import lru_cache

from psycopg_pool import ConnectionPool

from semantic_layer.config import settings


@lru_cache
def get_pool() -> ConnectionPool:
    return ConnectionPool(
        conninfo=settings.postgres_dsn,
        min_size=settings.pg_pool_min_size,
        max_size=settings.pg_pool_max_size,
        open=False,
    )


_pool_opened = False


def ensure_pool_open() -> None:
    """Open the cached pool exactly once per process. ConnectionPool.open() is itself
    idempotent (a redundant call is a harmless no-op in psycopg_pool 3.3.1), but the
    `_pool_opened` guard avoids re-invoking it on every query. Callable from web
    startup, the CLI, ingest, or sql_tools."""
    global _pool_opened
    if not _pool_opened:
        get_pool().open()
        _pool_opened = True
