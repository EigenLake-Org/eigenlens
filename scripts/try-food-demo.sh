#!/usr/bin/env bash
# Docker-only public demo. Run from any directory; no host Python installation needed.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mode="${1:-analyze}"
if [[ "$mode" != "analyze" && "$mode" != "agent" ]]; then
  echo "Usage: $0 [analyze|agent]" >&2
  exit 2
fi
if ! command -v docker >/dev/null || ! docker compose version >/dev/null 2>&1; then
  echo "Install Docker Desktop (or Docker Engine with Compose v2) first." >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "Start Docker, then rerun this command." >&2
  exit 1
fi
if [[ "$mode" == "agent" && ( -z "${OPENAI_API_KEY:-}" || -z "${OPENAI_MODEL:-}" ) ]]; then
  echo "Agent mode requires OPENAI_API_KEY and OPENAI_MODEL. Run without arguments for the no-key demo." >&2
  exit 1
fi
export VECTOR_ML_MCP_TOKEN="${VECTOR_ML_MCP_TOKEN:-local-demo-token}"
mkdir -p demo-output
# An explicit target list avoids starting unrelated local database profiles.
docker compose --profile food-demo up -d --build --wait --wait-timeout 180 qdrant service
docker compose --profile food-demo run --rm -T --interactive=false --build food-loader
docker compose run --rm -T --interactive=false --no-deps --user "$(id -u):$(id -g)" food-client \
  python -m examples.food_reviews "$mode" --cache-dir /cache --output /output
printf '\nReady. Reports: %s/demo-output\nQdrant dashboard: http://localhost:6333/dashboard\nMCP endpoint: http://localhost:8000/mcp\n' "$PWD"
echo "Rerun this script safely; review IDs are stable and the verified download is cached."
