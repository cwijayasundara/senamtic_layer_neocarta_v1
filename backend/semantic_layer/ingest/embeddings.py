"""Create embeddings + vector indexes for chunks and metadata nodes."""

import hashlib
import struct

from neo4j import Driver

from neocarta.enrichment.embeddings import OpenAIEmbeddingsConnector

from semantic_layer.config import settings
from semantic_layer.ingest.llm import get_openai_client


def fake_vector(text: str, dim: int) -> list[float]:
    """Deterministic pseudo-embedding from a text hash — no OpenAI call. For scale
    runs where exact semantic quality is not under test (routing is keyword-based)."""
    out: list[float] = []
    i = 0
    while len(out) < dim:
        digest = hashlib.sha256(f"{text}:{i}".encode()).digest()
        for j in range(0, len(digest), 4):
            if len(out) >= dim:
                break
            (val,) = struct.unpack("I", digest[j:j + 4])
            out.append((val / 0xFFFFFFFF) * 2.0 - 1.0)   # in [-1, 1]
        i += 1
    return out


def embed_chunks(driver: Driver, batch: int = 64) -> None:
    """Embed Chunk.text into Chunk.embedding and ensure a vector index exists."""
    if settings.fake_embeddings:
        with driver.session(database=settings.neo4j_database) as session:
            rows = session.run(
                "MATCH (c:Chunk) WHERE c.embedding IS NULL RETURN c.id AS id, c.text AS text"
            ).data()
            for i in range(0, len(rows), batch):
                window = rows[i:i + batch]
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (c:Chunk {id: row.id})
                    CALL db.create.setNodeVectorProperty(c, 'embedding', row.vec)
                    """,
                    rows=[{"id": w["id"], "vec": fake_vector(w["text"] or "", settings.embedding_dimensions)}
                          for w in window],
                )
        _ensure_chunk_vector_index(driver)
        return
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
    """Create a vector index named `chunk_embeddings` on Chunk.embedding.

    We deliberately own the index name (rather than NeoCarta's auto-generated
    one) so that search tools can query it by a stable name. Any pre-existing
    NeoCarta-named index on the same property is dropped first to avoid a
    duplicate-index error."""
    with driver.session(database=settings.neo4j_database) as session:
        session.run("DROP INDEX chunk_vector_index IF EXISTS")
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
    if settings.fake_embeddings:
        return  # routing uses keyword catalog search; skip costly metadata embeds
    connector = OpenAIEmbeddingsConnector(
        driver,
        client=get_openai_client(),
        embedding_model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        database_name=settings.neo4j_database,
    )
    connector.run()
