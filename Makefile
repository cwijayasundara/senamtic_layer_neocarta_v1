.PHONY: up down seed seed-postgres seed-sqlite test install serve-apis ingest

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
