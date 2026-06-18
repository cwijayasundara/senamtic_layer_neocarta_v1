"""Write Document and Chunk nodes to Neo4j (idempotent via MERGE)."""

from neo4j import Driver

from semantic_layer.config import settings

_DOC_CYPHER = """
MERGE (d:Document {id: $doc_id})
SET d.title = $title, d.path = $path, d.num_pages = $num_pages, d.file_hash = $file_hash
WITH d
UNWIND $chunks AS ch
MERGE (c:Chunk {id: ch.chunk_id})
SET c.text = ch.text, c.ordinal = ch.ordinal, c.doc_id = ch.doc_id
MERGE (d)-[:HAS_CHUNK]->(c)
"""


def load_document(driver: Driver, doc: dict) -> None:
    with driver.session(database=settings.neo4j_database) as session:
        session.run(
            _DOC_CYPHER,
            doc_id=doc["doc_id"], title=doc["title"], path=doc["path"],
            num_pages=doc["num_pages"], file_hash=doc.get("file_hash"), chunks=doc["chunks"],
        )
