from eval.scorer import routing_scores, check_answer


def test_routing_perfect():
    s = routing_scores(["table:a", "table:b"], ["table:a", "table:b"])
    assert s["precision"] == 1.0 and s["recall"] == 1.0 and s["hit"] is True


def test_routing_partial():
    s = routing_scores(["table:a", "table:x"], ["table:a", "table:b"])
    assert s["precision"] == 0.5
    assert s["recall"] == 0.5
    assert s["hit"] is False


def test_routing_empty_expected_is_pass():
    s = routing_scores(["table:a"], [])
    assert s["precision"] == 1.0 and s["recall"] == 1.0 and s["hit"] is True


def test_check_answer_contains():
    assert check_answer("Total revenue by REGION was $5M", {"type": "contains", "values": ["region", "revenue"]})
    assert not check_answer("no match here", {"type": "contains", "values": ["region"]})


def test_check_answer_numeric():
    assert check_answer("There are 5000 customers.", {"type": "numeric", "value": 5000, "tol": 0})
    assert check_answer("about 4998 rows", {"type": "numeric", "value": 5000, "tol": 5})
    assert not check_answer("about 4000 rows", {"type": "numeric", "value": 5000, "tol": 5})
