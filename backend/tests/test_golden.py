from eval.golden import load_golden, GoldenQuestion

VALID_CATEGORIES = {
    "single-table-agg", "multi-table-join", "cross-source-bridge",
    "document-rag", "period-filter", "near-miss-routing",
}


def test_golden_loads_and_is_well_formed():
    qs = load_golden()
    assert len(qs) >= 20
    assert all(isinstance(q, GoldenQuestion) for q in qs)


def test_every_question_has_required_fields():
    for q in load_golden():
        assert q.id and q.question
        assert q.category in VALID_CATEGORIES
        assert isinstance(q.expected_tables, list)
        assert q.answer_check.get("type") in {"contains", "numeric"}
        if q.category not in {"near-miss-routing", "document-rag"}:
            assert q.expected_tables, f"{q.id} needs expected_tables"


def test_table_ids_are_fully_qualified():
    for q in load_golden():
        for t in q.expected_tables:
            assert t.startswith("table:"), f"{q.id}: {t} not a table id"


def test_ids_are_unique():
    ids = [q.id for q in load_golden()]
    assert len(ids) == len(set(ids))
