import pytest

from semantic_layer.ingest.entities import _clean_entities, extract_entities, POLE_LABELS


def test_clean_entities_accepts_typed_entity():
    out = _clean_entities([
        {
            "name": "Blackwell",
            "base_type": "Object",
            "subtype": "ProductArchitecture",
            "confidence": 0.91,
            "evidence": "Blackwell architecture",
        }
    ])

    assert out == [
        {
            "name": "Blackwell",
            "label": "Object",
            "base_type": "Object",
            "subtype": "ProductArchitecture",
            "confidence": 0.91,
            "evidence": "Blackwell architecture",
        }
    ]


def test_clean_entities_degrades_low_confidence_subtype():
    out = _clean_entities([
        {
            "name": "Blackwell",
            "base_type": "Object",
            "subtype": "ProductArchitecture",
            "confidence": 0.79,
            "evidence": "Blackwell architecture",
        }
    ])

    assert out == [
        {
            "name": "Blackwell",
            "label": "Object",
            "base_type": "Object",
            "subtype": None,
            "confidence": 0.79,
            "evidence": "Blackwell architecture",
        }
    ]


def test_clean_entities_accepts_legacy_label_shape():
    out = _clean_entities([{"name": "NVIDIA", "label": "Org"}])

    assert out == [
        {
            "name": "NVIDIA",
            "label": "Org",
            "base_type": "Org",
            "subtype": None,
            "confidence": 1.0,
            "evidence": "",
        }
    ]


def test_clean_entities_drops_invalid_subtype_base_mismatch_and_clamps_confidence():
    out = _clean_entities([
        {
            "name": "NVIDIA",
            "base_type": "Organization",
            "subtype": "ProductArchitecture",
            "confidence": 1.5,
            "evidence": None,
        }
    ])

    assert out == [
        {
            "name": "NVIDIA",
            "label": "Org",
            "base_type": "Org",
            "subtype": None,
            "confidence": 1.0,
            "evidence": "",
        }
    ]


@pytest.mark.openai
def test_extract_entities_finds_nvidia_org(require_openai):
    text = (
        "NVIDIA announced record Data Center revenue driven by the Blackwell "
        "architecture. CEO Jensen Huang highlighted demand in the United States."
    )
    ents = extract_entities(text)
    assert len(ents) > 0
    assert all(e["label"] in POLE_LABELS for e in ents)
    assert all(e["base_type"] in POLE_LABELS for e in ents)
    assert all("subtype" in e for e in ents)
    names = {e["name"].lower() for e in ents}
    assert any("nvidia" in n for n in names)
