"""Generate business glossary terms from schema columns and tag them onto columns."""

import json

from neo4j import Driver

from neocarta.data_model.rdbms import BusinessTerm
from neocarta.ingest.rdbms import Neo4jRDBMSLoader

from semantic_layer.config import settings
from semantic_layer.ingest.llm import get_chat_model

_PROMPT = (
    "You are a data catalog expert for an NVIDIA enterprise. For each column below, "
    "produce a concise business term and a one-sentence business definition. "
    'Return ONLY a JSON array of {{"column_id","name","description"}}. Columns:\n\n{cols}'
)


def generate_business_terms(columns: list[dict]) -> list[dict]:
    model = get_chat_model()
    payload = "\n".join(f'- {c["column_id"]} (column "{c["name"]}" on table {c["table"]})' for c in columns)
    resp = model.invoke(_PROMPT.format(cols=payload))
    content = resp.content if hasattr(resp, "content") else str(resp)
    content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        return []
    valid_ids = {c["column_id"] for c in columns}
    return [
        {"column_id": t["column_id"], "name": t["name"].strip(), "description": t["description"].strip()}
        for t in raw
        if t.get("column_id") in valid_ids and t.get("name") and t.get("description")
    ]


def load_business_terms(driver: Driver, terms: list[dict]) -> None:
    """Create BusinessTerm nodes and TAGGED_WITH edges from columns to terms."""
    loader = Neo4jRDBMSLoader(driver, database_name=settings.neo4j_database)
    bt_nodes = [
        BusinessTerm(id=f"term:{i}", name=t["name"], description=t["description"])
        for i, t in enumerate(terms)
    ]
    if bt_nodes:
        loader.load_business_term_nodes(bt_nodes, overwrite_existing=True)
    with driver.session(database=settings.neo4j_database) as session:
        session.run(
            """
            UNWIND $rows AS row
            MATCH (bt:BusinessTerm {id: row.term_id})
            MATCH (c:Column {id: row.column_id})
            MERGE (c)-[:TAGGED_WITH]->(bt)
            """,
            rows=[{"term_id": f"term:{i}", "column_id": t["column_id"]} for i, t in enumerate(terms)],
        )
