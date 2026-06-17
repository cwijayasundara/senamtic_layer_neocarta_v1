from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "nvidia"
    postgres_user: str = "neocarta"
    postgres_password: str = "neocarta"

    sqlite_dir: str = "data/seed"

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
