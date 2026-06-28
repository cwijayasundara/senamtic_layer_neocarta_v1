"""Clean and load document fact triplets extracted from chunks."""

import hashlib
import json

from neo4j import Driver

from semantic_layer.config import settings
from semantic_layer.ingest.llm import get_chat_model


_FACT_PROMPT = (
    "Extract factual subject-predicate-object triplets from EACH numbered text below. "
    "Return ONLY a JSON array with one element per text, in the same order. Each element "
    "must be an array of objects with keys \"subject\", \"predicate\", \"object\", "
    "\"confidence\", \"valid_from\", and \"valid_until\". Use concise noun phrases for "
    "subjects and objects, and short verb phrases for predicates. confidence must be a "
    "number from 0.0 to 1.0. valid_from and valid_until should be strings when the text "
    "states a time bound, otherwise null. Deduplicate identical triplets within each text. "
    "Use an empty array for a text with no facts. Return exactly {n} elements.\n\n{body}"
)


def _confidence(value) -> float:
    if value is None:
        return 1.0
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, confidence))


def _clean_optional_string(value) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def clean_facts(raw) -> list[dict]:
    """Return valid, normalized fact triplets from LLM-like rows."""
    out, seen = [], set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        subject = (item.get("subject") or "").strip()
        predicate = (item.get("predicate") or "").strip()
        obj = (item.get("object") or "").strip()
        if not subject or not predicate or not obj:
            continue

        key = (subject.lower(), predicate.lower(), obj.lower())
        if key in seen:
            continue
        seen.add(key)

        text = f"{subject} / {predicate} / {obj}"
        out.append({
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "text": text,
            "confidence": _confidence(item.get("confidence")),
            "valid_from": _clean_optional_string(item.get("valid_from")),
            "valid_until": _clean_optional_string(item.get("valid_until")),
        })
    return out


def fact_id(chunk_id: str, subject: str, predicate: str, obj: str) -> str:
    """Stable Fact id from chunk id plus lowercased triplet content."""
    payload = json.dumps(
        {
            "chunk_id": chunk_id,
            "subject": subject.strip().lower(),
            "predicate": predicate.strip().lower(),
            "object": obj.strip().lower(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"fact:{digest}"


def load_facts(driver: Driver, chunk_id: str, facts: list[dict]) -> int:
    """MERGE Fact nodes and Chunk-[:HAS_FACT]->Fact edges for one chunk."""
    rows = [
        {
            "id": fact_id(chunk_id, fact["subject"], fact["predicate"], fact["object"]),
            "subject": fact["subject"],
            "predicate": fact["predicate"],
            "object": fact["object"],
            "text": fact["text"],
            "confidence": fact["confidence"],
            "source_chunk_id": chunk_id,
            "valid_from": fact.get("valid_from"),
            "valid_until": fact.get("valid_until"),
        }
        for fact in facts
        if (
            isinstance(fact, dict)
            and (fact.get("subject") or "").strip()
            and (fact.get("predicate") or "").strip()
            and (fact.get("object") or "").strip()
        )
    ]
    if not rows:
        return 0
    with driver.session(database=settings.neo4j_database) as session:
        session.run(
            """
            MATCH (c:Chunk {id: $chunk_id})
            UNWIND $rows AS row
            MERGE (f:Fact {id: row.id})
            SET f.subject = row.subject,
                f.predicate = row.predicate,
                f.object = row.object,
                f.text = row.text,
                f.confidence = row.confidence,
                f.source_chunk_id = row.source_chunk_id,
                f.valid_from = row.valid_from,
                f.valid_until = row.valid_until
            MERGE (c)-[:HAS_FACT]->(f)
            """,
            chunk_id=chunk_id,
            rows=rows,
        )
    return len(rows)


def extract_facts_batch(texts: list[str]) -> list[list[dict]]:
    """Extract cleaned fact triplets for many chunks in one LLM call."""
    if not texts:
        return []
    body = "\n\n".join(f"[{i}] {text[:6000]}" for i, text in enumerate(texts))
    model = get_chat_model()
    resp = model.invoke(_FACT_PROMPT.format(n=len(texts), body=body))
    content = resp.content if hasattr(resp, "content") else str(resp)
    content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        groups = json.loads(content)
    except json.JSONDecodeError:
        return [[] for _ in texts]
    if not isinstance(groups, list) or len(groups) != len(texts):
        return [[] for _ in texts]
    return [clean_facts(group) for group in groups]
