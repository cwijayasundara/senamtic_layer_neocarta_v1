from semantic_layer.eval.evalset import load_evalset


def test_load_evalset_default_has_expected_shape():
    items = load_evalset()
    assert len(items) >= 4
    for it in items:
        assert set(it) >= {"id", "question", "expect"}
        assert isinstance(it["question"], str) and it["question"]
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids))   # unique ids
