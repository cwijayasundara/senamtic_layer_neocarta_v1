import pytest

from semantic_layer.ingest.entities import extract_entities, POLE_LABELS


@pytest.mark.openai
def test_extract_entities_finds_nvidia_org(require_openai):
    text = (
        "NVIDIA announced record Data Center revenue driven by the Blackwell "
        "architecture. CEO Jensen Huang highlighted demand in the United States."
    )
    ents = extract_entities(text)
    assert len(ents) > 0
    assert all(e["label"] in POLE_LABELS for e in ents)
    names = {e["name"].lower() for e in ents}
    assert any("nvidia" in n for n in names)
