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
