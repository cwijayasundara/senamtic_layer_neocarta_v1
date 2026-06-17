"""The web path must always emit a final answer, even when the controller errors.

Robustness now lives in the controller (extract -> plan -> legs -> synthesize), which
stream_chat_events simply delegates to. This is a pure unit test — no services/key."""

from semantic_layer.web.events import stream_chat_events
from semantic_layer.agent import controller as ctrl


def test_stream_emits_answer_when_controller_errors(monkeypatch):
    def _boom(_q):
        raise RuntimeError("planner exploded")

    monkeypatch.setattr(ctrl, "extract_intent", _boom)
    out = list(stream_chat_events("q"))
    assert out[-1]["type"] == "answer"            # never leave the UI hanging
    assert "planner exploded" in out[-1]["content"]
    assert out[-1]["sql_runs"] == [] and out[-1]["caveats"] == []
