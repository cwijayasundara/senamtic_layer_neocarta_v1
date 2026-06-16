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
