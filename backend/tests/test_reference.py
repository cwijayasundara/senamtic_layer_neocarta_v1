from data.generators.reference import build_dimensions


def test_dimensions_have_expected_counts():
    d = build_dimensions()
    assert len(d["regions"]) == 4
    assert len(d["countries"]) == 9
    assert len(d["industries"]) == 6
    assert len(d["segments"]) == 5
    assert len(d["architectures"]) == 5
    assert len(d["product_lines"]) == 10
    assert len(d["fiscal_periods"]) == 8


def test_every_country_points_to_a_valid_region():
    d = build_dimensions()
    region_ids = {r["region_id"] for r in d["regions"]}
    assert all(c["region_id"] in region_ids for c in d["countries"])


def test_product_lines_reference_valid_segment_and_architecture():
    d = build_dimensions()
    seg_ids = {s["segment_id"] for s in d["segments"]}
    arch_ids = {a["architecture_id"] for a in d["architectures"]}
    for pl in d["product_lines"]:
        assert pl["segment_id"] in seg_ids
        assert pl["architecture_id"] in arch_ids


def test_blackwell_data_center_line_exists():
    d = build_dimensions()
    seg = {s["segment_id"]: s["name"] for s in d["segments"]}
    arch = {a["architecture_id"]: a["name"] for a in d["architectures"]}
    pairs = {(seg[pl["segment_id"]], arch[pl["architecture_id"]]) for pl in d["product_lines"]}
    assert ("Data Center", "Blackwell") in pairs
