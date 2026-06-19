#!/usr/bin/env bash
# NeoCarta-Local — provision infrastructure and data (part 1 of 2).
#
# Installs dependencies, starts the Dockerized data stores (Neo4j + Postgres),
# seeds the databases, and ingests the knowledge graph. It does NOT start the
# backend app — run ./start-backend.sh for that.
#
# Usage:  ./setup.sh [--scale]
# Then:   ./start-backend.sh   (start the mock APIs + agent web API)
#         ./start-ui.sh        (start the web UI)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
BACKEND="$ROOT/backend"
VENV="$BACKEND/.venv"
PY="$VENV/bin/python"

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

# --- arg parsing ------------------------------------------------------------
SCALE=false
for arg in "$@"; do
  case "$arg" in
    --scale) SCALE=true ;;
    -h|--help)
      printf "Usage: %s [--scale]\n" "$(basename "$0")"
      printf "  --scale   provision the scale catalog (1000 distractor tables + scaled core)\n"
      printf "            instead of the baseline core; needs OPENAI_API_KEY for routing embeddings\n"
      exit 0 ;;
    *) printf "Unknown option: %s (try --help)\n" "$arg" >&2; exit 2 ;;
  esac
done

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
if command -v uv >/dev/null 2>&1; then
  # uv is fast and installs into the venv without needing pip inside it.
  ( cd "$BACKEND" && uv pip install --quiet --python "$PY" -e ".[dev]" )
else
  # Some venvs (created with --without-pip, or populated by uv) have no pip.
  # Bootstrap it with the stdlib ensurepip before installing.
  if ! "$PY" -m pip --version >/dev/null 2>&1; then
    say "Bootstrapping pip into the virtualenv"
    "$PY" -m ensurepip --upgrade >/dev/null 2>&1 \
      || die "pip is missing from $VENV and could not be bootstrapped. Recreate it with: rm -rf '$VENV' && python3 -m venv '$VENV', or install uv (https://docs.astral.sh/uv/)."
  fi
  ( cd "$BACKEND" && "$PY" -m pip install --quiet --upgrade pip && "$PY" -m pip install --quiet -e ".[dev]" )
fi

# --- 5. seed the databases --------------------------------------------------
if [ "$SCALE" = "true" ]; then
  say "Seeding the SCALE catalog (core at scale volume + 1000 distractor tables) + SQLite"
  ( cd "$BACKEND" && SCALE_MODE=true "$PY" -m data.seed_scale )
  ( cd "$BACKEND" && "$PY" -m data.seed_sqlite )
else
  say "Seeding databases (Postgres sales schema + SQLite financials/org)"
  ( cd "$BACKEND" && "$PY" -m data.seed_postgres && "$PY" -m data.seed_sqlite )
fi

# --- 6. ingest the knowledge graph -----------------------------------------
if [ "$SCALE" = "true" ]; then
  if [ "$HAVE_KEY" = "1" ]; then
    say "Ingesting the SCALE catalog (1072 tables + table embeddings; schema routing on)"
    ( cd "$BACKEND" && SCALE_MODE=true SCHEMA_ROUTING_ENABLED=true FAKE_EMBEDDINGS=true \
        "$PY" -m semantic_layer.ingest.pipeline )
  else
    warn "No OPENAI_API_KEY — scale routing needs table embeddings; ingesting metadata only (keyword routing fallback)."
    ( cd "$BACKEND" && SCALE_MODE=true SCHEMA_ROUTING_ENABLED=true \
        "$PY" -c "from semantic_layer.ingest.pipeline import run_ingest; print(run_ingest(with_llm=False, reset=True))" )
  fi
elif [ "$HAVE_KEY" = "1" ]; then
  say "Ingesting the knowledge graph (metadata + documents + entities + glossary + embeddings)"
  ( cd "$BACKEND" && "$PY" -m semantic_layer.ingest.pipeline )
else
  say "Ingesting metadata + documents only (no OPENAI_API_KEY — entities/glossary/embeddings skipped)"
  ( cd "$BACKEND" && "$PY" -c "from semantic_layer.ingest.pipeline import run_ingest; print(run_ingest(with_llm=False, reset=True))" )
fi

# --- 7. done ----------------------------------------------------------------
say "Setup complete — infrastructure and data are ready:"
printf "  • Neo4j browser:   http://localhost:7474   (neo4j / neocarta123)\n"
printf "  • Postgres:        localhost:5432           (neocarta / neocarta / nvidia)\n"
printf "\nNext:  \033[1m./start-backend.sh\033[0m   (start the mock APIs + agent web API)\n"
printf "       \033[1m./start-ui.sh\033[0m        (start the web UI →  http://localhost:3005)\n"
printf "Stop:  kill \$(cat logs/*.pid) 2>/dev/null; docker compose down\n"
