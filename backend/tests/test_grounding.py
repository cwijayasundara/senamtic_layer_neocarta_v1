from semantic_layer.web.grounding import check_numeric_grounding


def _sql(rows, columns=("v",)):
    return [{"source": "sales_pg", "sql": "SELECT …", "columns": list(columns),
             "rows": rows, "row_count": len(rows), "error": None}]


def test_flags_number_absent_from_rows():
    caveats = check_numeric_grounding("Revenue was $999,999M.", _sql([[12345]]), [], [])
    assert len(caveats) == 1
    assert "999,999" in caveats[0]


def test_no_caveat_when_number_present():
    caveats = check_numeric_grounding("Total revenue was 1234567.", _sql([[1234567]]), [], [])
    assert caveats == []


def test_percentage_fraction_normalization():
    # prose says 38.81%, the cell stores the fraction 0.3881
    caveats = check_numeric_grounding("Blackwell share is 38.81%.", _sql([[0.3881]]), [], [])
    assert caveats == []


def test_magnitude_suffix_matches_raw_cell():
    # prose "$81,615M" == 81,615,000,000 in the cell
    caveats = check_numeric_grounding("Revenue $81,615M.", _sql([[81615000000]]), [], [])
    assert caveats == []


def test_years_and_small_counts_are_noise():
    caveats = check_numeric_grounding("In FY2027 we found 3 tables.", [], [], [])
    assert caveats == []


def test_grounded_by_api_data_or_doc_quote():
    api = [{"source": "itsm", "path": "/tickets", "params": {}, "status": 200,
            "row_count": 1, "data": [{"open": 8}]}]
    docs = [{"doc_id": "doc:x", "chunk_id": "doc:x:chunk:1",
             "quote": "Revenue $81,615M", "score": 0.9}]
    assert check_numeric_grounding("There are 8 open tickets.", [], api, []) == []
    assert check_numeric_grounding("Revenue was $81,615M.", [], [], docs) == []
