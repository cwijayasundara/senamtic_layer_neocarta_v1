"""Turn document chunks into a connected context graph.

Beyond Document-[:HAS_CHUNK]->Chunk, we attach the entities mentioned in each chunk
(Chunk-[:MENTIONS]->Entity) and BRIDGE those entities to the canonical value layer
(Entity-[:REFERS_TO]->Value) so a press-release mention of 'Blackwell' converges on
the same node as sales.architecture.name='Blackwell'. Entities with no catalog match
still enrich the graph as connected nodes.
"""

import re

from neo4j import Driver

from semantic_layer.config import settings
from semantic_layer.ingest.value_indexer import norm

# "Q1 Fiscal 2027", "Q1 FY27" — quarter adjacent to a fiscal-year marker.
_PERIOD = re.compile(r"Q([1-4])\s+(?:Fiscal|FY)\s*'?(\d{2,4})", re.IGNORECASE)
# Fallback: a word-quarter anywhere plus a fiscal year stated separately.
_WORD_Q = re.compile(r"\b(first|second|third|fourth)[\s-]quarter\b", re.IGNORECASE)
_FY = re.compile(r"(?:Fiscal(?:\s+year)?|FY)\s*'?(\d{2,4})", re.IGNORECASE)
_WORD_TO_Q = {"first": "Q1", "second": "Q2", "third": "Q3", "fourth": "Q4"}


def _norm_year(raw: str) -> int:
    """Expand a 2-digit fiscal year ('27') to four digits ('2027')."""
    return 2000 + int(raw) if len(raw) == 2 else int(raw)


def extract_period(doc: dict) -> dict | None:
    """Return {'fiscal_year', 'quarter'} for the period a document primarily reports.

    Scans the headline chunks (press releases state the period in the title/summary),
    preferring an explicit 'Q<n> Fiscal/FY <year>' over a word-quarter + fiscal year.
    Returns None when no period is found."""
    text = "\n".join(c.get("text", "") for c in doc.get("chunks", [])[:4])
    m = _PERIOD.search(text)
    if m:
        return {"fiscal_year": _norm_year(m.group(2)), "quarter": f"Q{m.group(1)}"}
    wq, fy = _WORD_Q.search(text), _FY.search(text)
    if wq and fy:
        return {"fiscal_year": _norm_year(fy.group(1)), "quarter": _WORD_TO_Q[wq.group(1).lower()]}
    return None


def period_key(fiscal_year: int, quarter: str) -> str:
    """Canonical :Period MERGE key, e.g. 'FY2027-Q1'."""
    return f"FY{fiscal_year}-{quarter}"


def link_document_period(driver: Driver, doc_id: str, period: dict | None) -> None:
    """Link a Document to the fiscal Period it reports via COVERS_PERIOD.

    MERGEs on the canonical key, so a doc-covered period attaches to the SQL-backed
    Period node when one exists (preserving its fiscal_period_id) instead of duplicating."""
    if not period:
        return
    key = period_key(period["fiscal_year"], period["quarter"])
    with driver.session(database=settings.neo4j_database) as session:
        session.run(
            """
            MATCH (d:Document {id: $doc_id})
            MERGE (p:Period {key: $key})
              ON CREATE SET p.fiscal_year = $fy, p.quarter = $q
            MERGE (d)-[:COVERS_PERIOD]->(p)
            """,
            doc_id=doc_id, key=key, fy=period["fiscal_year"], q=period["quarter"],
        )


def load_entities(driver: Driver, chunk_id: str, entities: list[dict]) -> None:
    """MERGE Entity {norm} nodes and Chunk-[:MENTIONS]->Entity edges for one chunk."""
    rows = [
        {"name": e["name"], "label": e["label"], "norm": norm(e["name"])}
        for e in entities if (e.get("name") or "").strip()
    ]
    if not rows:
        return
    with driver.session(database=settings.neo4j_database) as session:
        session.run(
            """
            MATCH (c:Chunk {id: $chunk_id})
            UNWIND $rows AS row
            MERGE (e:Entity {norm: row.norm})
              ON CREATE SET e.name = row.name
              SET e.label = row.label
            MERGE (c)-[:MENTIONS]->(e)
            """,
            chunk_id=chunk_id, rows=rows,
        )


# Tokens too generic or too short to anchor a bridge on their own.
_BRIDGE_STOPWORDS = {"and", "the", "of", "for", "a", "an", "to", "in", "on", "at"}
# A single-token value must be at least this long to bridge, so region/quarter codes
# ('us', 'q1', 'ae') don't latch onto incidental tokens in document text.
_MIN_BRIDGE_TOKEN = 4


def _token_match(entity_norm: str, value_norm: str) -> bool:
    """True when a catalog value names a thing the entity also names, by whole words.

    A single-word value must appear as a full token in the entity (length-guarded,
    no stopwords); a multi-word value must appear as a contiguous token run. Whole-word
    matching avoids substring false positives ('india' inside 'indiana jones')."""
    e_tokens = entity_norm.split()
    v_tokens = value_norm.split()
    if not e_tokens or not v_tokens:
        return False
    if len(v_tokens) == 1:
        tok = v_tokens[0]
        if len(tok) < _MIN_BRIDGE_TOKEN or tok in _BRIDGE_STOPWORDS:
            return False
        return tok in e_tokens
    return any(
        e_tokens[i:i + len(v_tokens)] == v_tokens
        for i in range(len(e_tokens) - len(v_tokens) + 1)
    )


def bridge_entities_to_values(driver: Driver) -> int:
    """Link document Entities to catalog Values that name the same thing.

    Bridges on exact norm equality plus whole-word token overlap (_token_match), so
    'NVIDIA Blackwell GPUs' converges on the 'Blackwell' value. Returns the total number
    of Entity-[:REFERS_TO]->Value edges after bridging."""
    with driver.session(database=settings.neo4j_database) as session:
        entities = [r["norm"] for r in session.run("MATCH (e:Entity) RETURN DISTINCT e.norm AS norm")]
        values = [r["norm"] for r in session.run("MATCH (v:Value) RETURN DISTINCT v.norm AS norm")]
        pairs = [
            {"e": en, "v": vn}
            for en in entities for vn in values
            if en == vn or _token_match(en, vn)
        ]
        if pairs:
            session.run(
                """
                UNWIND $pairs AS pair
                MATCH (e:Entity {norm: pair.e}), (v:Value {norm: pair.v})
                MERGE (e)-[:REFERS_TO]->(v)
                """,
                pairs=pairs,
            )
        return session.run(
            "MATCH (:Entity)-[r:REFERS_TO]->(:Value) RETURN count(r) AS c"
        ).single()["c"]
