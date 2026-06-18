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


from semantic_layer.eval import run as run_mod
from semantic_layer.eval.run import run_eval


def test_run_eval_aggregates_scores():
    evalset = [
        {"id": "a", "question": "q1", "expect": "e1"},
        {"id": "b", "question": "q2", "expect": "e2"},
        {"id": "c", "question": "q3", "expect": "e3"},
    ]
    answers = {"q1": "A1", "q2": "A2", "q3": "A3"}
    scores = {"q1": 4, "q2": 3, "q3": 1}
    report = run_eval(
        evalset,
        answer_fn=lambda q: answers[q],
        judge_fn=lambda question, answer, expect: {"score": scores[question], "reason": "r"},
    )
    assert [r["id"] for r in report["results"]] == ["a", "b", "c"]
    assert report["mean_score"] == round((4 + 3 + 1) / 3, 2)
    assert report["pass_rate"] == round(2 / 3, 2)   # a & b pass (>=3), c fails


def test_default_answer_fn_extracts_final_answer(monkeypatch):
    events = [
        {"type": "tool_result", "scope": "sql", "content": "{}"},
        {"type": "answer", "content": "the final answer", "highlight": []},
    ]
    monkeypatch.setattr(run_mod, "answer_stream", lambda q: iter(events))
    assert run_mod.default_answer_fn("anything") == "the final answer"
