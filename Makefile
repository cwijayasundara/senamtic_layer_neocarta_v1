.PHONY: up down seed seed-postgres seed-sqlite test install serve-apis ingest ask serve-web scale-seed scale-ingest eval eval-baseline scale-teardown

install:
	cd backend && pip install -e ".[dev]"

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
