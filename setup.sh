#!/usr/bin/env bash
# NeoCarta-Local — one-shot platform setup.
#
# Installs dependencies, starts the Dockerized data stores (Neo4j + Postgres),
# seeds the databases, ingests the knowledge graph, and launches the backend
# simulators (mock enterprise APIs) and the agent web API.
#
# Usage:  ./setup.sh
# Then:   ./start-ui.sh   (to launch the web UI)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
BACKEND="$ROOT/backend"
VENV="$BACKEND/.venv"
PY="$VENV/bin/python"
UVICORN="$VENV/bin/uvicorn"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

say()  { printf "\n\033[1;32m==>\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33mWARNING:\033[0m %s\n" "$*"; }
die()  { printf "\n\033[1;31mERROR:\033[0m %s\n" "$*" >&2; exit 1; }

# --- 1. prerequisites -------------------------------------------------------
say "Checking prerequisites"
command -v docker  >/dev/null || die "Docker is required (install Docker Desktop)."
docker info >/dev/null 2>&1   || die "The Docker daemon is not running — start Docker Desktop."
command -v python3 >/dev/null || die "Python 3.11+ is required."

# --- 2. environment file ----------------------------------------------------
if [ ! -e "$BACKEND/.env" ]; then
  if [ -f "$ROOT/.env" ]; then
    ln -sf ../.env "$BACKEND/.env"; say "Linked backend/.env -> ../.env"
  else
    cp "$BACKEND/.env.example" "$BACKEND/.env"; say "Created backend/.env from example"
  fi
fi
if grep -Eq 'OPENAI_API_KEY=.{10,}' "$BACKEND/.env" 2>/dev/null; then
  HAVE_KEY=1
else
  HAVE_KEY=0
  warn "OPENAI_API_KEY is not set in backend/.env (or ./.env)."
  warn "The agent and document search require it. Add it and re-run for the full experience:"
  warn '  echo "OPENAI_API_KEY=sk-..." >> backend/.env'
fi

# --- 3. Docker data stores --------------------------------------------------
say "Starting Docker services (Neo4j + Postgres)"
docker compose up -d
printf "Waiting for Postgres"
until docker inspect --format '{{.State.Health.Status}}' neocarta-postgres 2>/dev/null | grep -q healthy; do printf "."; sleep 2; done
printf " ready\n"
printf "Waiting for Neo4j"
until docker inspect --format '{{.State.Health.Status}}' neocarta-neo4j 2>/dev/null | grep -q healthy; do printf "."; sleep 2; done
printf " ready\n"

# --- 4. Python venv + dependencies -----------------------------------------
if [ ! -x "$PY" ]; then say "Creating virtualenv at backend/.venv"; python3 -m venv "$VENV"; fi
say "Installing backend dependencies"
( cd "$BACKEND" && "$PY" -m pip install --quiet --upgrade pip && "$PY" -m pip install --quiet -e ".[dev]" )

# --- 5. seed the databases --------------------------------------------------
say "Seeding databases (Postgres sales schema + SQLite financials/org)"
( cd "$BACKEND" && "$PY" -m data.seed_postgres && "$PY" -m data.seed_sqlite )

# --- 6. ingest the knowledge graph -----------------------------------------
if [ "$HAVE_KEY" = "1" ]; then
  say "Ingesting the knowledge graph (metadata + documents + entities + glossary + embeddings)"
  ( cd "$BACKEND" && "$PY" -m semantic_layer.ingest.pipeline )
else
  say "Ingesting metadata + documents only (no OPENAI_API_KEY — entities/glossary/embeddings skipped)"
  ( cd "$BACKEND" && "$PY" -c "from semantic_layer.ingest.pipeline import run_ingest; print(run_ingest(with_llm=False, reset=True))" )
fi

# --- 7. launch backend services --------------------------------------------
start_service() {  # name  module:app  port
  local name="$1" target="$2" port="$3"
  if lsof -ti tcp:"$port" >/dev/null 2>&1; then
    warn "$name: port $port already in use — leaving the existing process."
    return
  fi
  ( cd "$BACKEND" && nohup "$UVICORN" "$target" --port "$port" > "$LOGDIR/$name.log" 2>&1 & echo $! > "$LOGDIR/$name.pid" )
  say "Started $name on :$port  (logs/$name.log)"
}
start_service "mock-apis" "semantic_layer.apis.app:app" 8001
start_service "web-api"   "semantic_layer.web.app:app"  8000

sleep 2
say "Platform is up:"
printf "  • Neo4j browser:   http://localhost:7474   (neo4j / neocarta123)\n"
printf "  • Postgres:        localhost:5432           (neocarta / neocarta / nvidia)\n"
printf "  • Mock APIs:       http://localhost:8001/docs\n"
printf "  • Agent web API:   http://localhost:8000     (GET /graph, GET /sources, POST /chat)\n"
printf "\nNext:  \033[1m./start-ui.sh\033[0m   →  http://localhost:3000\n"
printf "Stop:  kill \$(cat logs/*.pid) 2>/dev/null; docker compose down\n"
