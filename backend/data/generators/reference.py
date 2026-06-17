"""Curated NVIDIA-themed dimension data with deterministic integer ids."""

REGIONS = ["North America", "EMEA", "APAC", "LATAM"]

# (name, iso_code, region_name)
COUNTRIES = [
    ("United States", "US", "North America"),
    ("Canada", "CA", "North America"),
    ("United Kingdom", "GB", "EMEA"),
    ("Germany", "DE", "EMEA"),
    ("United Arab Emirates", "AE", "EMEA"),
    ("Japan", "JP", "APAC"),
    ("South Korea", "KR", "APAC"),
    ("Singapore", "SG", "APAC"),
    ("Brazil", "BR", "LATAM"),
]

INDUSTRIES = [
    "Cloud Service Provider",
    "Automotive",
    "Healthcare",
    "Gaming",
    "Research",
    "Financial Services",
]

SEGMENTS = [
    "Data Center",
    "Gaming",
    "Professional Visualization",
    "Automotive",
    "OEM & Other",
]

# (name, launch_year)
ARCHITECTURES = [
    ("Ampere", 2020),
    ("Ada Lovelace", 2022),
    ("Hopper", 2022),
    ("Grace", 2023),
    ("Blackwell", 2024),
]

# (name, segment_name, architecture_name)
PRODUCT_LINES = [
    ("H100", "Data Center", "Hopper"),
    ("H200", "Data Center", "Hopper"),
    ("B200", "Data Center", "Blackwell"),
    ("GB200", "Data Center", "Blackwell"),
    ("A100", "Data Center", "Ampere"),
    ("RTX 4090", "Gaming", "Ada Lovelace"),
    ("RTX 4080", "Gaming", "Ada Lovelace"),
    ("RTX 5090", "Gaming", "Blackwell"),
    ("RTX 6000 Ada", "Professional Visualization", "Ada Lovelace"),
    ("DRIVE Thor", "Automotive", "Blackwell"),
]

# NVIDIA fiscal year ends late January; FY label = the calendar year it ends in.
# (fiscal_year, quarter, start_date, end_date)
FISCAL_PERIODS = [
    (2024, "Q1", "2023-02-01", "2023-04-30"),
    (2024, "Q2", "2023-05-01", "2023-07-31"),
    (2024, "Q3", "2023-08-01", "2023-10-31"),
    (2024, "Q4", "2023-11-01", "2024-01-28"),
    (2025, "Q1", "2024-01-29", "2024-04-28"),
    (2025, "Q2", "2024-04-29", "2024-07-28"),
    (2025, "Q3", "2024-07-29", "2024-10-27"),
    (2025, "Q4", "2024-10-28", "2025-01-26"),
    (2026, "Q1", "2025-01-27", "2025-04-27"),
    (2026, "Q2", "2025-04-28", "2025-07-27"),
    (2026, "Q3", "2025-07-28", "2025-10-26"),
    (2026, "Q4", "2025-10-27", "2026-01-25"),
    (2027, "Q1", "2026-01-26", "2026-04-26"),
]


def build_dimensions():
    """Return all dimension rows with assigned integer primary keys and FKs."""
    regions = [{"region_id": i + 1, "name": n} for i, n in enumerate(REGIONS)]
    region_id = {r["name"]: r["region_id"] for r in regions}

    countries = [
        {"country_id": i + 1, "name": n, "iso_code": iso, "region_id": region_id[reg]}
        for i, (n, iso, reg) in enumerate(COUNTRIES)
    ]

    industries = [{"industry_id": i + 1, "name": n} for i, n in enumerate(INDUSTRIES)]
    segments = [{"segment_id": i + 1, "name": n} for i, n in enumerate(SEGMENTS)]
    architectures = [
        {"architecture_id": i + 1, "name": n, "launch_year": y}
        for i, (n, y) in enumerate(ARCHITECTURES)
    ]

    seg_id = {s["name"]: s["segment_id"] for s in segments}
    arch_id = {a["name"]: a["architecture_id"] for a in architectures}
    product_lines = [
        {
            "product_line_id": i + 1,
            "name": n,
            "segment_id": seg_id[seg],
            "architecture_id": arch_id[arch],
        }
        for i, (n, seg, arch) in enumerate(PRODUCT_LINES)
    ]

    fiscal_periods = [
        {
            "fiscal_period_id": i + 1,
            "fiscal_year": fy,
            "quarter": q,
            "start_date": s,
            "end_date": e,
        }
        for i, (fy, q, s, e) in enumerate(FISCAL_PERIODS)
    ]

    return {
        "regions": regions,
        "countries": countries,
        "industries": industries,
        "segments": segments,
        "architectures": architectures,
        "product_lines": product_lines,
        "fiscal_periods": fiscal_periods,
    }
