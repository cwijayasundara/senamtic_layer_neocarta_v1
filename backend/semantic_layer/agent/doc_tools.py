"""Document retrieval tool: vector search over chunk embeddings."""

import json

from langchain_core.tools import tool

from semantic_layer.agent.driver import driver
from semantic_layer.config import settings
from semantic_layer.ingest.embeddings import embed_query


@tool
def search_documents(query: str, k: int = 5) -> str:
    """Search the ingested documents (NVIDIA press releases) for relevant passages.

    Embeds the query and runs vector search over document chunks. Returns the top-k
    passages with their document id and similarity score, for citing in answers."""
    vec = embed_query(query)
    records = driver().execute_query(
        """
        CALL db.index.vector.queryNodes('chunk_embeddings', $k, $vec)
        YIELD node, score
        RETURN node.id AS chunk_id, node.doc_id AS doc_id,
               node.text AS text, score ORDER BY score DESC
        """,
        k=k, vec=vec, database_=settings.neo4j_database,
    ).records
    return json.dumps([
        {"chunk_id": r["chunk_id"], "doc_id": r["doc_id"],
         "text": r["text"], "score": r["score"]}
        for r in records
    ])
