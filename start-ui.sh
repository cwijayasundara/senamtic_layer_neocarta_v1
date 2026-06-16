#!/usr/bin/env bash
# NeoCarta-Local — start the Next.js web UI.
#
# Requires the backend web API (port 8000) to be running first — run ./setup.sh.
#
# Usage:  ./start-ui.sh            # runs on http://localhost:3005
#         ./start-ui.sh 3010       # or any port you like
#         UI_PORT=3010 ./start-ui.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND="$ROOT/frontend"
UI_PORT="${UI_PORT:-${1:-3005}}"

command -v node >/dev/null || { echo "ERROR: Node.js 20+ is required." >&2; exit 1; }

cd "$FRONTEND"
[ -f .env.local ] || { cp .env.local.example .env.local; echo "==> Created frontend/.env.local"; }
[ -d node_modules ] || { echo "==> Installing frontend dependencies (first run)"; npm install; }

# Free the chosen UI port if something is already listening on it.
if command -v lsof >/dev/null; then
  PIDS_ON_PORT="$(lsof -ti tcp:"$UI_PORT" 2>/dev/null || true)"
  if [ -n "$PIDS_ON_PORT" ]; then
    echo "==> Port $UI_PORT is in use — stopping process(es): $PIDS_ON_PORT"
    # shellcheck disable=SC2086
    kill $PIDS_ON_PORT 2>/dev/null || true
    sleep 1
  fi
fi

# Stop any leftover `next dev` server started from THIS frontend directory, so
# Next.js' one-dev-server-per-project lock doesn't reject a fresh start.
if command -v pgrep >/dev/null && command -v lsof >/dev/null; then
  for pid in $(pgrep -f "next.* dev" 2>/dev/null || true); do
    if lsof -a -p "$pid" -d cwd 2>/dev/null | grep -q "$FRONTEND"; then
      echo "==> Stopping a leftover Next dev server for this project (pid $pid)"
      kill "$pid" 2>/dev/null || true
    fi
  done
  sleep 1
fi

# Friendly heads-up if the backend API isn't reachable yet.
if command -v curl >/dev/null && ! curl -sf http://localhost:8000/health >/dev/null 2>&1; then
  echo "WARNING: the web API on http://localhost:8000 is not responding."
  echo "         Run ./setup.sh first, or the UI will load with an empty graph."
fi

echo "==> Starting the UI on http://localhost:$UI_PORT  (Ctrl+C to stop)"
PORT="$UI_PORT" npm run dev
