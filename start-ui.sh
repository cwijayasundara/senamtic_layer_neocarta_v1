#!/usr/bin/env bash
# NeoCarta-Local — start the Next.js web UI on http://localhost:3000.
#
# Requires the backend web API (port 8000) to be running first — run ./setup.sh.
#
# Usage:  ./start-ui.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND="$ROOT/frontend"

command -v node >/dev/null || { echo "ERROR: Node.js 20+ is required." >&2; exit 1; }

cd "$FRONTEND"
[ -f .env.local ] || { cp .env.local.example .env.local; echo "==> Created frontend/.env.local"; }
[ -d node_modules ] || { echo "==> Installing frontend dependencies (first run)"; npm install; }

# Friendly heads-up if the backend API isn't reachable yet.
if command -v curl >/dev/null && ! curl -sf http://localhost:8000/health >/dev/null 2>&1; then
  echo "WARNING: the web API on http://localhost:8000 is not responding."
  echo "         Run ./setup.sh first, or the UI will load with an empty graph."
fi

echo "==> Starting the UI on http://localhost:3000  (Ctrl+C to stop)"
npm run dev
