from semantic_layer.config import settings


def test_model_tiering_defaults_fall_back_to_llm_model():
    # Unset planner/synthesis models fall back to the mechanical llm_model.
    assert settings.planner_model_resolved == (settings.planner_model or settings.llm_model)
    assert settings.synthesis_model_resolved == (
        settings.synthesis_model or settings.planner_model or settings.llm_model)
    # With nothing set, both resolve to llm_model.
    if not settings.planner_model and not settings.synthesis_model:
        assert settings.planner_model_resolved == settings.llm_model
        assert settings.synthesis_model_resolved == settings.llm_model


def test_get_chat_model_accepts_model_override(monkeypatch):
    import semantic_layer.ingest.llm as llm_mod
    calls = []
    monkeypatch.setattr(llm_mod, "init_chat_model", lambda m: calls.append(m))
    llm_mod.get_chat_model("openai:some-model")  # explicit override
    llm_mod.get_chat_model()                     # default path
    assert calls == ["openai:some-model", llm_mod.settings.llm_model]


import pytest

from semantic_layer.agent.planner import Intent, extract_intent


def test_extract_intent_uses_planner_model_and_returns_intent(monkeypatch):
    import semantic_layer.agent.planner as planner_mod
    canned = Intent(terms=["EMEA", "Blackwell"], needs_sql=True, needs_doc=True)
    captured = {}

    class _FakeStructured:
        def invoke(self, _messages):
            return canned

    class _FakeModel:
        def with_structured_output(self, schema):
            captured["schema"] = schema
            return _FakeStructured()

    def _fake_get_chat_model(model=None):
        captured["model"] = model
        return _FakeModel()

    monkeypatch.setattr(planner_mod, "get_chat_model", _fake_get_chat_model)
    out = extract_intent("which EMEA Blackwell things, per the press release?")
    assert out is canned
    assert captured["schema"] is Intent
    assert captured["model"] == planner_mod.settings.planner_model_resolved


@pytest.mark.openai
def test_extract_intent_live_pulls_terms(require_openai):
    out = extract_intent(
        "In FY2025, which EMEA Cloud customers bought Blackwell Data Center products?")
    lowered = [t.lower() for t in out.terms]
    assert any("emea" in t for t in lowered)
    assert any("blackwell" in t for t in lowered)
    assert out.needs_sql is True
