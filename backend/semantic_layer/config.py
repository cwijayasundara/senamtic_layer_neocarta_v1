from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "nvidia"
    postgres_user: str = "neocarta"
    postgres_password: str = "neocarta"

    sqlite_dir: str = "data/seed"
    # BigQuery-audit-log-shaped query log mined by NeoCarta's query_log connector to
    # surface empirically-observed joins (OBSERVED_JOIN). Optional: ingest skips it
    # when the file is absent.
    query_log_file: str = "data/seed/query_log_sales.json"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neocarta123"

    openai_api_key: str = ""

    random_seed: int = 42

    neo4j_database: str = "neo4j"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    llm_model: str = "openai:gpt-5.4-mini"
    # The powerful model for intent extraction (and synthesis); falls back to
    # llm_model when unset. Set e.g. planner_model=openai:gpt-5.5 in .env to opt in.
    planner_model: str | None = None
    synthesis_model: str | None = None
    docs_dir: str = "../docs"
    agent_max_rows: int = 100
    # LangGraph superstep cap. Heavy multi-subagent questions need well above the
    # default 25; otherwise the run raises GraphRecursionError partway through.
    agent_recursion_limit: int = 100

    # Schema routing: retrieve-then-rank table selection for many-table scale.
    # Off by default so the deterministic sales planner and golden tests are
    # unchanged; turn on once the catalog grows past a handful of tables.
    schema_routing_enabled: bool = False
    schema_routing_k_ret: int = 20
    schema_routing_k_rank: int = 8

    # Entity-extraction throughput: chunks per LLM call, and concurrent in-flight
    # batches during ingest (replaces the old one-call-per-chunk serial loop).
    entity_batch_size: int = 10
    ingest_max_workers: int = 8

    # Incremental ingestion: skip re-parsing/re-embedding documents whose source
    # bytes are unchanged (only consulted when run_ingest is called with reset=False).
    ingest_skip_unchanged: bool = True

    # Query cache: exact-match + semantic (embedding cosine) lookup in front of the
    # agent, to skip re-running legs/synthesis for repeated or near-identical asks.
    query_cache_enabled: bool = True
    cache_max_entries: int = 512
    cache_ttl_seconds: int = 3600
    cache_similarity_threshold: float = 0.95

    # Postgres connection pool (replaces per-call psycopg.connect in sql_tools).
    # NOTE (multi-worker): the pool and the answer-concurrency gate are PER PROCESS.
    # Under N uvicorn/gunicorn workers the real Postgres connection ceiling is
    # N * pg_pool_max_size — size it against Postgres `max_connections`, and treat
    # max_concurrent_answers as a per-worker limit (cluster ceiling = N * value).
    pg_pool_min_size: int = 1
    pg_pool_max_size: int = 8

    # Controller concurrency: max simultaneous answers computed server-wide, and
    # the per-answer leg thread-pool size.
    max_concurrent_answers: int = 8
    leg_max_workers: int = 4

    # Bounded /graph rendering: cap the document chunk layer so the UI payload stays
    # renderable at thousands of documents (the structured layer is returned in full).
    graph_max_chunks: int = 300

    # Enterprise APIs to register/serve (comma-separated). Adding an API is config, not
    # code: list its name here and mount/serve its OpenAPI spec under /{name}.
    api_sources: str = "crm,itsm,partner,dgx"

    @property
    def api_source_list(self) -> list[str]:
        return [s.strip() for s in self.api_sources.split(",") if s.strip()]

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def planner_model_resolved(self) -> str:
        return self.planner_model or self.llm_model

    @property
    def synthesis_model_resolved(self) -> str:
        return self.synthesis_model or self.planner_model or self.llm_model


settings = Settings()
