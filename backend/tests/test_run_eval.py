# backend/tests/test_run_eval.py
from eval.golden import GoldenQuestion
from eval.run_eval import evaluate


def _q(id, expected, check, cat="single-table-agg"):
    return GoldenQuestion(id=id, question="q?", expected_tables=expected,
                          answer_check=check, category=cat)


def test_evaluate_aggregates_metrics():
    questions = [
        _q("a", ["table:x"], {"type": "contains", "values": ["ok"]}),
        _q("b", ["table:y"], {"type": "contains", "values": ["zzz"]}),
    ]
    route_fn = lambda q: {"a": ["table:x"], "b": ["table:w"]}[q.id]
    answer_fn = lambda q: {"a": "ok answer", "b": "wrong"}[q.id]
    clock = iter([0.0, 0.1, 0.2, 1.0, 1.1, 1.2])  # 3 timer() calls per question
    card = evaluate(route_fn, answer_fn, questions, timer=lambda: next(clock))

    assert card["summary"]["n"] == 2
    assert card["summary"]["routing_hit_rate"] == 0.5   # only "a" hits
    assert card["summary"]["answer_accuracy"] == 0.5    # only "a" passes
    by_id = {r["id"]: r for r in card["questions"]}
    assert by_id["a"]["answer_ok"] is True
    assert by_id["b"]["routing"]["hit"] is False
