import json

from semantic_layer.ingest import entities as ent_mod
from semantic_layer.ingest.entities import extract_entities_batch


class _Resp:
    def __init__(self, content):
        self.content = content


class _FakeModel:
    def __init__(self, content):
        self._content = content

    def invoke(self, _prompt):
        return _Resp(self._content)


def test_extract_entities_batch_groups_per_chunk(monkeypatch):
    payload = json.dumps([
        [{"name": "NVIDIA", "label": "Org"}, {"name": "nvidia", "label": "Org"}],  # dup dropped
        [{"name": "Jensen Huang", "label": "Person"}, {"name": "Bob", "label": "Alien"}],  # bad label dropped
    ])
    monkeypatch.setattr(ent_mod, "get_chat_model", lambda model=None: _FakeModel(payload))
    out = extract_entities_batch(["chunk about NVIDIA", "chunk about the CEO"])
    assert len(out) == 2
    assert out[0] == [{"name": "NVIDIA", "label": "Org"}]
    assert out[1] == [{"name": "Jensen Huang", "label": "Person"}]


def test_extract_entities_batch_handles_bad_json(monkeypatch):
    monkeypatch.setattr(ent_mod, "get_chat_model", lambda model=None: _FakeModel("not json"))
    out = extract_entities_batch(["a", "b", "c"])
    assert out == [[], [], []]


def test_extract_entities_batch_count_mismatch(monkeypatch):
    # LLM returns 2 groups for 3 input texts -> all empty per spec.
    payload = json.dumps([
        [{"name": "NVIDIA", "label": "Org"}],
        [{"name": "Jensen Huang", "label": "Person"}],
    ])
    monkeypatch.setattr(ent_mod, "get_chat_model", lambda model=None: _FakeModel(payload))
    out = extract_entities_batch(["a", "b", "c"])
    assert out == [[], [], []]


def test_extract_entities_batch_empty_input(monkeypatch):
    monkeypatch.setattr(ent_mod, "get_chat_model",
                        lambda model=None: (_ for _ in ()).throw(AssertionError("should not call model")))
    assert extract_entities_batch([]) == []
