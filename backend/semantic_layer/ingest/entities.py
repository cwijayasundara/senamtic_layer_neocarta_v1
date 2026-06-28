"""Extract POLE+O entities (Person, Org, Location, Event, Object) from text via LLM."""

import json

from semantic_layer.ingest.llm import get_chat_model
from semantic_layer.ingest.ontology import subtype_base_map

POLE_LABELS = {"Person", "Org", "Location", "Event", "Object"}
SUBTYPE_CONFIDENCE_THRESHOLD = 0.80

_SUBTYPE_BASE_MAP = subtype_base_map()
_LABEL_ALIASES = {"Organization": "Org"}
_LABEL_BY_LOWER = {label.lower(): label for label in POLE_LABELS}
_LABEL_BY_LOWER["organization"] = "Org"


def _subtype_catalog_prompt() -> str:
    grouped = {label: [] for label in sorted(POLE_LABELS)}
    for subtype, base_type in sorted(_SUBTYPE_BASE_MAP.items()):
        grouped[base_type].append(subtype)
    return "\n".join(f"- {base_type}: {', '.join(subtypes) or 'none'}" for base_type, subtypes in grouped.items())


_SUBTYPE_CATALOG_PROMPT = _subtype_catalog_prompt()

_PROMPT = (
    "Extract named entities from the text. Return ONLY a JSON array of objects "
    'with keys "name", "base_type", "subtype", "confidence", and "evidence". '
    "base_type must be one of: Person, Org, Location, Event, Object. "
    "subtype must be null or one of the catalog values for that base_type. "
    "confidence must be a number from 0.0 to 1.0 for the subtype assignment. "
    "evidence should be a short quote or phrase from the text. Deduplicate by name.\n\n"
    "Subtype catalog:\n{subtype_catalog}\n\nText:\n\n{text}"
)


def extract_entities(text: str) -> list[dict]:
    model = get_chat_model()
    resp = model.invoke(_PROMPT.format(subtype_catalog=_SUBTYPE_CATALOG_PROMPT, text=text[:6000]))
    content = resp.content if hasattr(resp, "content") else str(resp)
    content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        return []
    return _clean_entities(raw)


_BATCH_PROMPT = (
    "Extract named entities from EACH numbered text below. Return ONLY a JSON array "
    "with one element per text, in the same order. Each element is an array of objects "
    'with keys "name", "base_type", "subtype", "confidence", and "evidence". '
    "base_type must be one of: Person, Org, Location, Event, Object. subtype must be "
    "null or one of the catalog values for that base_type. confidence must be a number "
    "from 0.0 to 1.0 for the subtype assignment. evidence should be a short quote or "
    "phrase from the text. Deduplicate by name within each text. Use an empty array for "
    "a text with no entities. Return exactly {n} elements.\n\n"
    "Subtype catalog:\n{subtype_catalog}\n\n{body}"
)


def _normalize_base_type(value) -> str | None:
    if not isinstance(value, str):
        return None
    value = _LABEL_ALIASES.get(value.strip(), value.strip())
    return _LABEL_BY_LOWER.get(value.lower())


def _clean_confidence(value) -> float:
    if value is None:
        return 1.0
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, confidence))


def _clean_entities(raw: list) -> list[dict]:
    out, seen = [], set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip()
        base_type = _normalize_base_type(item.get("base_type") or item.get("label"))
        name_key = name.lower()
        if not name or base_type not in POLE_LABELS or name_key in seen:
            continue

        confidence = _clean_confidence(item.get("confidence"))
        subtype = item.get("subtype")
        if not isinstance(subtype, str):
            subtype = None
        else:
            subtype = subtype.strip() or None
        if (
            subtype
            and (_SUBTYPE_BASE_MAP.get(subtype) != base_type or confidence < SUBTYPE_CONFIDENCE_THRESHOLD)
        ):
            subtype = None

        evidence = item.get("evidence")
        if not isinstance(evidence, str):
            evidence = ""

        seen.add(name_key)
        out.append({
            "name": name,
            "label": base_type,
            "base_type": base_type,
            "subtype": subtype,
            "confidence": confidence,
            "evidence": evidence,
        })
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
    resp = model.invoke(_BATCH_PROMPT.format(n=len(texts), subtype_catalog=_SUBTYPE_CATALOG_PROMPT, body=body))
    content = resp.content if hasattr(resp, "content") else str(resp)
    content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        groups = json.loads(content)
    except json.JSONDecodeError:
        return [[] for _ in texts]
    if not isinstance(groups, list) or len(groups) != len(texts):
        return [[] for _ in texts]
    return [_clean_entities(g) for g in groups]
