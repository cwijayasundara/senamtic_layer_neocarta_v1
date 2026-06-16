from data.generators.org import generate_org


def test_departments_and_locations_present():
    data = generate_org(seed=42)
    assert len(data["departments"]) == 6
    assert len(data["locations"]) == 5


def test_headcount_references_valid_dimensions():
    data = generate_org(seed=42)
    dept_ids = {d["department_id"] for d in data["departments"]}
    loc_ids = {l["location_id"] for l in data["locations"]}
    for h in data["headcount"]:
        assert h["department_id"] in dept_ids
        assert h["location_id"] in loc_ids
        assert h["employee_count"] > 0


def test_deterministic():
    assert generate_org(seed=42) == generate_org(seed=42)
