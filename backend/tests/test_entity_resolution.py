from semantic_layer.ingest.doc_graph import _candidate_pairs, _token_match


def _bruteforce(entities, values):
    return {(en, vn) for en in entities for vn in values if en == vn or _token_match(en, vn)}


def test_candidate_pairs_equals_bruteforce_on_sample():
    entities = ["nvidia blackwell gpus", "data center revenue", "jensen huang",
                "emea cloud customers", "blackwell"]
    values = ["blackwell", "data center", "cloud service provider", "emea", "hopper",
              "data center", "q1"]
    got = {(p["e"], p["v"]) for p in _candidate_pairs(entities, values)}
    assert got == _bruteforce(entities, values)


def test_candidate_pairs_no_duplicates():
    pairs = _candidate_pairs(["blackwell blackwell", "blackwell"], ["blackwell"])
    keys = [(p["e"], p["v"]) for p in pairs]
    assert len(keys) == len(set(keys))


def test_candidate_pairs_empty_inputs():
    assert _candidate_pairs([], ["x"]) == []
    assert _candidate_pairs(["x"], []) == []
