"""Deterministic quarterly income statement and daily stock prices."""

import random
from datetime import date, timedelta

from .reference import FISCAL_PERIODS


def generate_financials(seed=42):
    rng = random.Random(seed)

    income_statement = []
    pid = 0
    revenue = 7000.0  # $M, grows each quarter
    for fy, q, start, end in FISCAL_PERIODS:
        pid += 1
        revenue = round(revenue * rng.uniform(1.05, 1.30), 1)
        gross_margin_pct = round(rng.uniform(60.0, 75.0), 1)
        operating_income = round(revenue * rng.uniform(0.30, 0.55), 1)
        net_income = round(operating_income * rng.uniform(0.80, 0.95), 1)
        eps = round(net_income / 2470.0, 2)  # ~2.47B diluted shares
        income_statement.append(
            {
                "period_id": pid,
                "fiscal_year": fy,
                "quarter": q,
                "revenue": revenue,
                "gross_margin_pct": gross_margin_pct,
                "operating_income": operating_income,
                "net_income": net_income,
                "eps": eps,
            }
        )

    # daily stock prices across the whole fiscal range
    start_date = date.fromisoformat(FISCAL_PERIODS[0][2])
    end_date = date.fromisoformat(FISCAL_PERIODS[-1][3])
    stock_prices = []
    price = 22.0
    d = start_date
    while d <= end_date:
        if d.weekday() < 5:  # weekdays only
            change = rng.uniform(-0.04, 0.05)
            open_p = round(price, 2)
            close_p = round(price * (1 + change), 2)
            high_p = round(max(open_p, close_p) * rng.uniform(1.0, 1.03), 2)
            low_p = round(min(open_p, close_p) * rng.uniform(0.97, 1.0), 2)
            stock_prices.append(
                {
                    "price_date": d.isoformat(),
                    "open": open_p,
                    "high": high_p,
                    "low": low_p,
                    "close": close_p,
                    "volume": rng.randint(200_000_000, 600_000_000),
                }
            )
            price = close_p
        d += timedelta(days=1)

    return {"income_statement": income_statement, "stock_prices": stock_prices}
