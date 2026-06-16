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

## Mock enterprise APIs (Plan 2)

Four NVIDIA-themed mock REST APIs run as one Uvicorn process (mounted sub-apps),
serving deterministic in-memory data. Accounts reuse the sales customer base, so
API records join back to the `sales` data by `account_id` (== `customer_id`).

```bash
source backend/.venv/bin/activate
make serve-apis     # uvicorn on http://localhost:8001
```

| API | Prefix | Key endpoints | OpenAPI |
|-----|--------|---------------|---------|
| CRM | `/crm` | `/accounts`, `/contacts`, `/opportunities` | `/crm/openapi.json` |
| Support/ITSM | `/itsm` | `/tickets`, `/rma` | `/itsm/openapi.json` |
| Partner inventory | `/partner` | `/partners`, `/inventory` | `/partner/openapi.json` |
| DGX Cloud telemetry | `/dgx` | `/usage` | `/dgx/openapi.json` |

Health check: `GET /health`. These OpenAPI specs are what Plan 3's NeoCarta API
extractor introspects into the semantic graph as virtual tables/columns.
