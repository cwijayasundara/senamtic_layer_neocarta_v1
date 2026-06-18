"""Extract POLE+O entities (Person, Org, Location, Event, Object) from text via LLM."""

import json

from semantic_layer.ingest.llm import get_chat_model

POLE_LABELS = {"Person", "Org", "Location", "Event", "Object"}

_PROMPT = (
    "Extract named entities from the text. Return ONLY a JSON array of objects "
    'with keys "name" and "label". label must be one of: '
    "Person, Org, Location, Event, Object. Deduplicate by name. Text:\n\n{text}"
)


def extract_entities(text: str) -> list[dict]:
    model = get_chat_model()
    resp = model.invoke(_PROMPT.format(text=text[:6000]))
    content = resp.content if hasattr(resp, "content") else str(resp)
    content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        return []
    out = []
    seen = set()
    for item in raw:
        name = (item.get("name") or "").strip()
        label = (item.get("label") or "").strip().capitalize()
        if name and label in POLE_LABELS and name.lower() not in seen:
            seen.add(name.lower())
            out.append({"name": name, "label": label})
    return out


_BATCH_PROMPT = (
    "Extract named entities from EACH numbered text below. Return ONLY a JSON array "
    "with one element per text, in the same order. Each element is an array of objects "
    'with keys "name" and "label"; label must be one of: Person, Org, Location, Event, '
    "Object. Deduplicate by name within each text. Use an empty array for a text with no "
    "entities. Return exactly {n} elements.\n\n{body}"
)


def _clean_entities(raw: list) -> list[dict]:
    out, seen = [], set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip()
        label = (item.get("label") or "").strip().capitalize()
        if name and label in POLE_LABELS and name.lower() not in seen:
            seen.add(name.lower())
            out.append({"name": name, "label": label})
    return out


def extract_entities_batch(texts: list[str]) -> list[list[dict]]:
    """Extract POLE+O entities for many chunks in ONE LLM call.

    Returns one entity list per input text, in the same order. On any parse error
    or element-count mismatch, returns empty lists for every text (the caller can
    retry the batch smaller or fall back to per-chunk extraction)."""
    if not texts:
        return []
    body = "\n\n".join(f"[{i}] {t[:6000]}" for i, t in enumerate(texts))
    model = get_chat_model()
    resp = model.invoke(_BATCH_PROMPT.format(n=len(texts), body=body))
    content = resp.content if hasattr(resp, "content") else str(resp)
    content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        groups = json.loads(content)
    except json.JSONDecodeError:
        return [[] for _ in texts]
    if not isinstance(groups, list) or len(groups) != len(texts):
        return [[] for _ in texts]
    return [_clean_entities(g) for g in groups]
