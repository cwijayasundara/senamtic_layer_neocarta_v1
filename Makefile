.PHONY: up down seed seed-postgres seed-sqlite test install serve-apis ingest ask serve-web scale-seed scale-ingest eval eval-baseline scale-teardown setup setup-baseline deps

install:
	cd backend && pip install -e ".[dev]"

# One-command first-time provisioning (Docker stores + Python venv & deps + data
# + graph ingest). Uses backend/.venv explicitly so it works without an activated
# venv. Default provisions the full SCALE catalog (~1000 distractor tables + scaled
# core); needs OPENAI_API_KEY in backend/.env for routing embeddings.
# Use `make setup-baseline` for the small core-only demo (40 customers, ~16 tables).
# Create backend/.venv if missing and install deps into it. Robust to a pip-less
# venv (e.g. one created by `uv venv`): prefer uv, else bootstrap pip via ensurepip.
deps:
	cd backend && test -x .venv/bin/python || python3 -m venv .venv
	cd backend && if command -v uv >/dev/null 2>&1; then \
		uv pip install --quiet --python .venv/bin/python -e ".[dev]"; \
	else \
		.venv/bin/python -m pip --version >/dev/null 2>&1 || .venv/bin/python -m ensurepip --upgrade >/dev/null 2>&1 \
			|| { echo "ERROR: .venv has no pip and ensurepip failed; install uv or recreate: rm -rf backend/.venv && python3 -m venv backend/.venv" >&2; exit 1; }; \
		.venv/bin/python -m pip install --quiet --upgrade pip && .venv/bin/python -m pip install --quiet -e ".[dev]"; \
	fi

setup: up deps
	cd backend && SCALE_MODE=true .venv/bin/python -m data.seed_scale
	cd backend && .venv/bin/python -m data.seed_sqlite
	cd backend && SCALE_MODE=true SCHEMA_ROUTING_ENABLED=true FAKE_EMBEDDINGS=true .venv/bin/python -m semantic_layer.ingest.pipeline
	@echo "==> Setup complete (scale catalog: ~1000 tables). Next: ./start-backend.sh  then  ./start-ui.sh"

setup-baseline: up deps
	cd backend && .venv/bin/python -m data.seed_postgres && .venv/bin/python -m data.seed_sqlite
	cd backend && .venv/bin/python -m semantic_layer.ingest.pipeline
	@echo "==> Baseline setup complete (~16 tables). Next: ./start-backend.sh  then  ./start-ui.sh"

up:
	docker compose up -d
	@echo "Waiting for Postgres to be healthy..."
	@until docker inspect --format '{{.State.Health.Status}}' neocarta-postgres | grep -q healthy; do sleep 2; done
	@echo "Postgres ready."

down:
	docker compose down

seed-postgres:
	cd backend && python -m data.seed_postgres

seed-sqlite:
	cd backend && python -m data.seed_sqlite

seed: seed-postgres seed-sqlite

test:
	cd backend && python -m pytest -v

serve-apis:
	cd backend && uvicorn semantic_layer.apis.app:app --port 8001 --reload

ingest:
	cd backend && python -m semantic_layer.ingest.pipeline

ask:
	cd backend && python -m semantic_layer.agent.cli "$(q)"

serve-web:
	cd backend && uvicorn semantic_layer.web.app:app --port 8000 --reload

# --- scale / agent-performance harness ---
# Seed the answerable core at scale volume + create empty distractor tables.
scale-seed:
	cd backend && SCALE_MODE=true python -m data.seed_scale

# Ingest in scale mode (distractor schemas + synthetic APIs), routing on, fake embeds.
scale-ingest:
	cd backend && SCALE_MODE=true SCHEMA_ROUTING_ENABLED=true FAKE_EMBEDDINGS=true \
		python -m semantic_layer.ingest.pipeline

# Score the agent over the golden set against the current (scaled) graph.
eval:
	cd backend && SCHEMA_ROUTING_ENABLED=true python -m eval.run_eval --out scorecard.json

# Baseline score against the default small catalog (routing off) for comparison.
eval-baseline:
	cd backend && python -m eval.run_eval --out scorecard-baseline.json

# Drop all scale_* schemas, restoring the baseline DB.
scale-teardown:
	cd backend && python -c "from data.seed_scale import drop_scale_schemas; from semantic_layer.config import settings; drop_scale_schemas(settings.postgres_dsn); print('scale schemas dropped')"
