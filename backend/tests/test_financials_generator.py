from data.generators.financials import generate_financials


def test_income_statement_one_row_per_fiscal_quarter():
    data = generate_financials(seed=42)
    keys = {(r["fiscal_year"], r["quarter"]) for r in data["income_statement"]}
    assert len(keys) == len(data["income_statement"]) == 8


def test_net_income_not_above_revenue():
    data = generate_financials(seed=42)
    for r in data["income_statement"]:
        assert 0 < r["net_income"] <= r["revenue"]


def test_stock_prices_are_ohlc_consistent():
    data = generate_financials(seed=42)
    assert len(data["stock_prices"]) > 0
    for p in data["stock_prices"]:
        assert p["low"] <= p["open"] <= p["high"]
        assert p["low"] <= p["close"] <= p["high"]


def test_deterministic():
    assert generate_financials(seed=42) == generate_financials(seed=42)
