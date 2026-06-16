"""Deterministic org / headcount data for the org.db SQLite source."""

import random

DEPARTMENTS = [
    "Research & Development",
    "Sales & Marketing",
    "Operations",
    "Finance",
    "Legal",
    "Information Technology",
]

# (city, country, region)
LOCATIONS = [
    ("Santa Clara", "United States", "North America"),
    ("Austin", "United States", "North America"),
    ("Tel Aviv", "Israel", "EMEA"),
    ("Bangalore", "India", "APAC"),
    ("Taipei", "Taiwan", "APAC"),
]

_FISCAL = [(2024, q) for q in ("Q1", "Q2", "Q3", "Q4")] + [
    (2025, q) for q in ("Q1", "Q2", "Q3", "Q4")
]


def generate_org(seed=42):
    rng = random.Random(seed)

    departments = [
        {"department_id": i + 1, "name": n} for i, n in enumerate(DEPARTMENTS)
    ]
    locations = [
        {"location_id": i + 1, "city": c, "country": co, "region": reg}
        for i, (c, co, reg) in enumerate(LOCATIONS)
    ]

    headcount = []
    sid = 0
    for dept in departments:
        for loc in locations:
            base = rng.randint(50, 1200)
            for fy, q in _FISCAL:
                sid += 1
                base = int(base * rng.uniform(1.0, 1.08))  # grows over time
                headcount.append(
                    {
                        "snapshot_id": sid,
                        "department_id": dept["department_id"],
                        "location_id": loc["location_id"],
                        "fiscal_year": fy,
                        "quarter": q,
                        "employee_count": base,
                    }
                )

    return {"departments": departments, "locations": locations, "headcount": headcount}
