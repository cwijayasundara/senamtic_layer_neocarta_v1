#!/usr/bin/env bash
# NeoCarta-Local — start the backend app (part 2 of 2).
#
# Launches the mock enterprise APIs (:8001) and the agent web API (:8000).
# Run ./setup.sh first to provision Docker, dependencies, and the ingested graph.
#
# Usage:  ./start-backend.sh            # foreground: stream logs; Ctrl-C stops both servers
#         ./start-backend.sh -d         # detached: launch in the background, return to the prompt
# Then:   ./start-ui.sh   (to launch the web UI — use a second terminal, or run with -d above)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# --- options ----------------------------------------------------------------
# Default is foreground/follow. -d/--detach restores the old background behavior;
# -f/--follow is kept as an accepted no-op so existing muscle memory still works.
FOLLOW=1
for arg in "$@"; do
  case "$arg" in
    -d|--detach) FOLLOW=0 ;;
    -f|--follow) FOLLOW=1 ;;
    -h|--help)
      printf "Usage: %s [-d|--detach]\n" "$(basename "$0")"
      printf "  (default)        run in the foreground: stream both API logs; Ctrl-C stops both servers\n"
      printf "  -d, --detach     launch in the background and return to the prompt\n"
      exit 0 ;;
    *) printf "Unknown option: %s (try --help)\n" "$arg" >&2; exit 2 ;;
  esac
done
BACKEND="$ROOT/backend"
UVICORN="$BACKEND/.venv/bin/uvicorn"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

say()  { printf "\n\033[1;32m==>\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33mWARNING:\033[0m %s\n" "$*"; }
die()  { printf "\n\033[1;31mERROR:\033[0m %s\n" "$*" >&2; exit 1; }

# --- prerequisites ----------------------------------------------------------
[ -x "$UVICORN" ] || die "Backend dependencies are not installed — run ./setup.sh first."
if command -v docker >/dev/null; then
  docker inspect --format '{{.State.Health.Status}}' neocarta-neo4j 2>/dev/null | grep -q healthy \
    || warn "Neo4j container is not healthy — run ./setup.sh first (the web API needs it)."
fi

# --- launch services --------------------------------------------------------
# Free a port by stopping whatever is listening on it (TERM, then KILL).
free_port() {
  local port="$1" pids
  command -v lsof >/dev/null || return 0
  pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
  [ -n "$pids" ] || return 0
  say "Port $port in use — stopping the existing process(es): $pids"
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  for _ in $(seq 1 10); do
    lsof -ti tcp:"$port" >/dev/null 2>&1 || return 0
    sleep 0.5
  done
  pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
    sleep 1
  fi
}

start_service() {  # name  module:app  port
  local name="$1" target="$2" port="$3" pid
  free_port "$port"
  # `exec` replaces the backgrounded subshell with uvicorn itself, so $! is the
  # uvicorn PID (not a wrapper shell). That keeps logs/<name>.pid accurate, so the
  # documented `kill $(cat logs/*.pid)` actually stops the server instead of
  # orphaning it and leaving the port held.
  ( cd "$BACKEND" && exec nohup "$UVICORN" "$target" --port "$port" > "$LOGDIR/$name.log" 2>&1 ) &
  pid=$!
  echo "$pid" > "$LOGDIR/$name.pid"
  disown "$pid" 2>/dev/null || true
  say "Started $name on :$port  (pid $pid, logs/$name.log)"
}
# Stop both servers, freeing their ports. Used by follow-mode's Ctrl-C trap.
stop_servers() {
  printf "\n"
  say "Shutting down backend..."
  local pids p still
  pids="$(cat "$LOGDIR"/*.pid 2>/dev/null || true)"
  # shellcheck disable=SC2086
  [ -n "$pids" ] && kill $pids 2>/dev/null || true
  for _ in $(seq 1 6); do
    still=0
    for p in $pids; do kill -0 "$p" 2>/dev/null && still=1; done
    [ "$still" -eq 0 ] && break
    sleep 0.5
  done
  # shellcheck disable=SC2086
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
  rm -f "$LOGDIR"/*.pid
  say "Backend stopped. (Neo4j/Postgres still running — 'docker compose down' to stop them.)"
  exit 0
}

start_service "mock-apis" "semantic_layer.apis.app:app" 8001
start_service "web-api"   "semantic_layer.web.app:app"  8000

# In follow mode, arm the Ctrl-C trap NOW — before the health-wait below — so an
# interrupt during startup still tears the (detached) servers down instead of
# orphaning them.
[ "$FOLLOW" -eq 1 ] && trap stop_servers INT TERM

# Wait for the web API to answer before claiming the backend is up.
if command -v curl >/dev/null; then
  printf "Waiting for the web API"
  for _ in $(seq 1 20); do
    curl -sf http://localhost:8000/health >/dev/null 2>&1 && break
    printf "."; sleep 0.5
  done
  curl -sf http://localhost:8000/health >/dev/null 2>&1 \
    && printf " ready\n" \
    || warn "web-api did not respond on :8000 yet — check logs/web-api.log"
else
  sleep 2
fi

say "Backend is up:"
printf "  • Mock APIs:       http://localhost:8001/docs\n"
printf "  • Agent web API:   http://localhost:8000     (GET /graph, GET /sources, POST /chat)\n"

# --- follow mode: stream logs here, Ctrl-C shuts both servers down ----------
if [ "$FOLLOW" -eq 1 ]; then
  printf "\nFollowing logs — press \033[1mCtrl-C\033[0m to stop both servers.\n\n"
  # tail in the background so the INT trap (armed above) fires promptly, then
  # wait on it. `wait` returns when the signal arrives and stop_servers runs.
  tail -n +1 -f "$LOGDIR/mock-apis.log" "$LOGDIR/web-api.log" &
  TAIL_PID=$!
  wait "$TAIL_PID"
else
  printf "\nNext:  \033[1m./start-ui.sh\033[0m   →  http://localhost:3005\n"
  printf "Stop:  kill \$(cat logs/*.pid) 2>/dev/null; docker compose down\n"
fi
