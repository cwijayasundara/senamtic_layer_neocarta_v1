# NeoCarta-Local — Backend (Plan 1: Data Foundation)

## Prerequisites
- Docker + Docker Compose
- Python 3.11+

## Setup
```bash
cp backend/.env.example backend/.env        # adjust if needed
make up                                      # start postgres + neo4j

python3 -m venv backend/.venv                # create the virtualenv
source backend/.venv/bin/activate            # activate it (Makefile targets need this active)
make install                                 # pip install -e ".[dev]"

make seed                                    # load postgres sales schema + sqlite databases
make test                                    # run the full test suite
```

> The Makefile's `test`, `seed`, and `install` targets call bare `python`/`pip`,
> so the `backend/.venv` virtualenv must be **activated** first (the `source ...`
> line above). Without it those targets use system Python and will fail to import deps.

## What this provides
- Postgres `sales` schema: 11 normalized tables supporting 6+-table joins.
- SQLite `financials.db` (income statement + stock prices) and `org.db` (headcount).
- Deterministic, reproducible synthetic NVIDIA-themed data (seeded RNG).

## Verify the deep join
After `make seed`, the 11-table join in `tests/test_seed_postgres.py`
(`test_deep_eleven_table_join_returns_rows`) confirms a question like
"Data Center revenue for Blackwell products" resolves across
order_line → product → product_line → segment/architecture →
sales_order → fiscal_period → customer → industry → country → region.

## Next plans
2. Mock enterprise APIs · 3. Graph ingestion · 4. deepagents agent · 5. Web app.
