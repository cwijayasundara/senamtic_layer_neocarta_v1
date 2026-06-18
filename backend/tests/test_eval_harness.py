from semantic_layer.eval.evalset import load_evalset


def test_load_evalset_default_has_expected_shape():
    items = load_evalset()
    assert len(items) >= 4
    for it in items:
        assert set(it) >= {"id", "question", "expect"}
        assert isinstance(it["question"], str) and it["question"]
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids))   # unique ids


from semantic_layer.eval import judge as judge_mod
from semantic_layer.eval.judge import judge_answer


class _FakeStructured:
    def __init__(self, value):
        self._value = value

    def invoke(self, _messages):
        return self._value


class _FakeModel:
    def __init__(self, value):
        self._value = value

    def with_structured_output(self, _schema, **_kwargs):
        return _FakeStructured(self._value)


def test_judge_answer_returns_score_and_reason(monkeypatch):
    verdict = judge_mod._Verdict(score=4, reason="Names Data Center, cites sales DB.")
    monkeypatch.setattr(judge_mod, "get_chat_model", lambda model=None: _FakeModel(verdict))
    out = judge_answer("Which segment leads revenue?",
                       "Data Center leads, per the sales database.",
                       "Names Data Center as highest-revenue segment from sales SQL.")
    assert out["score"] == 4
    assert "Data Center" in out["reason"]
