"""Create embeddings + vector indexes for chunks and metadata nodes."""

from neo4j import Driver

import neocarta.ingest.indexes as nc_indexes
from neocarta.enrichment.embeddings import OpenAIEmbeddingsConnector

from semantic_layer.config import settings
from semantic_layer.ingest.llm import get_openai_client


def embed_chunks(driver: Driver, batch: int = 64) -> None:
    """Embed Chunk.text into Chunk.embedding and ensure a vector index exists."""
    client = get_openai_client()
    with driver.session(database=settings.neo4j_database) as session:
        rows = session.run(
            "MATCH (c:Chunk) WHERE c.embedding IS NULL RETURN c.id AS id, c.text AS text"
        ).data()
        for i in range(0, len(rows), batch):
            window = rows[i : i + batch]
            vectors = client.embeddings.create(
                model=settings.embedding_model,
                input=[r["text"] for r in window],
                dimensions=settings.embedding_dimensions,
            ).data
            session.run(
                """
                UNWIND $rows AS row
                MATCH (c:Chunk {id: row.id})
                CALL db.create.setNodeVectorProperty(c, 'embedding', row.vec)
                """,
                rows=[{"id": w["id"], "vec": v.embedding} for w, v in zip(window, vectors)],
            )
    _ensure_chunk_vector_index(driver)


def _ensure_chunk_vector_index(driver: Driver) -> None:
    try:
        nc_indexes.create_vector_index(
            driver,
            node_label="Chunk",
            dimensions=settings.embedding_dimensions,
            database_name=settings.neo4j_database,
        )
    except Exception:
        # Fallback to raw Cypher if NeoCarta's helper targets a different property.
        with driver.session(database=settings.neo4j_database) as session:
            session.run(
                f"""
                CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {{indexConfig: {{
                  `vector.dimensions`: {settings.embedding_dimensions},
                  `vector.similarity_function`: 'cosine'
                }}}}
                """
            )


def embed_metadata_nodes(driver: Driver) -> None:
    """Embed Table/Column/BusinessTerm nodes via NeoCarta's OpenAI connector."""
    connector = OpenAIEmbeddingsConnector(
        driver,
        client=get_openai_client(),
        embedding_model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        database_name=settings.neo4j_database,
    )
    connector.run()
