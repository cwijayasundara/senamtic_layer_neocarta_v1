"""Create embeddings + vector indexes for chunks and metadata nodes."""

import hashlib
import struct

from neo4j import Driver

from semantic_layer.config import settings
from semantic_layer.ingest.llm import get_openai_client
from semantic_layer.ingest.table_descriptions import TABLE_DESCRIPTIONS


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


def embed_query(text: str) -> list[float]:
    """Embed a single query string to a vector for query-time vector search.
    Shared by document and table retrieval so the call lives in one place."""
    return get_openai_client().embeddings.create(
        model=settings.embedding_model, input=[text],
        dimensions=settings.embedding_dimensions,
    ).data[0].embedding


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


def embed_facts(driver: Driver, batch: int = 64) -> None:
    """Embed Fact.text into Fact.embedding and ensure a vector index exists."""
    if settings.fake_embeddings:
        with driver.session(database=settings.neo4j_database) as session:
            rows = session.run(
                "MATCH (f:Fact) WHERE f.embedding IS NULL RETURN f.id AS id, f.text AS text"
            ).data()
            for i in range(0, len(rows), batch):
                window = rows[i:i + batch]
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (f:Fact {id: row.id})
                    CALL db.create.setNodeVectorProperty(f, 'embedding', row.vec)
                    """,
                    rows=[{"id": w["id"], "vec": fake_vector(w["text"] or "", settings.embedding_dimensions)}
                          for w in window],
                )
        _ensure_fact_vector_index(driver)
        return
    client = get_openai_client()
    with driver.session(database=settings.neo4j_database) as session:
        rows = session.run(
            "MATCH (f:Fact) WHERE f.embedding IS NULL RETURN f.id AS id, f.text AS text"
        ).data()
        for i in range(0, len(rows), batch):
            window = rows[i:i + batch]
            vectors = client.embeddings.create(
                model=settings.embedding_model,
                input=[r["text"] for r in window],
                dimensions=settings.embedding_dimensions,
            ).data
            session.run(
                """
                UNWIND $rows AS row
                MATCH (f:Fact {id: row.id})
                CALL db.create.setNodeVectorProperty(f, 'embedding', row.vec)
                """,
                rows=[{"id": w["id"], "vec": v.embedding} for w, v in zip(window, vectors)],
            )
    _ensure_fact_vector_index(driver)


def _ensure_fact_vector_index(driver: Driver) -> None:
    """Create a vector index named `fact_embeddings` on Fact.embedding."""
    with driver.session(database=settings.neo4j_database) as session:
        session.run("DROP INDEX fact_vector_index IF EXISTS")
        session.run(
            f"""
            CREATE VECTOR INDEX fact_embeddings IF NOT EXISTS
            FOR (f:Fact) ON (f.embedding)
            OPTIONS {{indexConfig: {{
              `vector.dimensions`: {settings.embedding_dimensions},
              `vector.similarity_function`: 'cosine'
            }}}}
            """
        )


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


def _table_embed_text(name: str, cols: list[str], description: str = "") -> str:
    """Text embedded per table: name, an optional curated description, plus column
    names. With description='' the output is the prior name+columns form."""
    parts = [name]
    if description:
        parts.append(description)
    if cols:
        parts.append(f"columns: {', '.join(cols)}")
    return " — ".join(parts)


def embed_tables(driver: Driver, batch: int = 64) -> None:
    """Embed each Table from its name + column names into Table.embedding and ensure
    the `table_embeddings` vector index exists.

    Always real (unlike embed_chunks it ignores fake_embeddings): schema routing
    retrieves over these vectors, so they must carry real semantics."""
    client = get_openai_client()
    with driver.session(database=settings.neo4j_database) as session:
        rows = session.run(
            """
            MATCH (t:Table)
            OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
            WITH t, collect(c.name) AS cols
            RETURN t.id AS id, t.name AS name, cols
            """
        ).data()
        for i in range(0, len(rows), batch):
            window = rows[i:i + batch]
            texts = [
                _table_embed_text(r["name"], r["cols"], TABLE_DESCRIPTIONS.get(r["id"], ""))
                for r in window
            ]
            vectors = client.embeddings.create(
                model=settings.embedding_model, input=texts,
                dimensions=settings.embedding_dimensions,
            ).data
            session.run(
                """
                UNWIND $rows AS row
                MATCH (t:Table {id: row.id})
                CALL db.create.setNodeVectorProperty(t, 'embedding', row.vec)
                """,
                rows=[{"id": w["id"], "vec": v.embedding} for w, v in zip(window, vectors)],
            )
    _ensure_table_vector_index(driver)


def _ensure_table_vector_index(driver: Driver) -> None:
    """Create a vector index named `table_embeddings` on Table.embedding — a stable
    name we own (mirroring `chunk_embeddings`). Drop any NeoCarta-named index on the
    same property first to avoid a duplicate-index error."""
    with driver.session(database=settings.neo4j_database) as session:
        session.run("DROP INDEX table_vector_index IF EXISTS")
        session.run(
            f"""
            CREATE VECTOR INDEX table_embeddings IF NOT EXISTS
            FOR (t:Table) ON (t.embedding)
            OPTIONS {{indexConfig: {{
              `vector.dimensions`: {settings.embedding_dimensions},
              `vector.similarity_function`: 'cosine'
            }}}}
            """
        )
