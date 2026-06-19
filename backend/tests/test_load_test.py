from eval.golden import GoldenQuestion
from eval.load_test import run_load


def test_run_load_counts_and_errors():
    qs = [GoldenQuestion(id=str(i), question="q", expected_tables=[],
                         answer_check={"type": "contains", "values": []},
                         category="single-table-agg") for i in range(4)]
    calls = {"n": 0}

    def send_fn(_q):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")

    clock = iter([float(i) for i in range(100)])
    res = run_load(send_fn, qs, concurrency=2, rounds=1, timer=lambda: next(clock))
    assert res["n"] == 4
    assert res["errors"] == 1
