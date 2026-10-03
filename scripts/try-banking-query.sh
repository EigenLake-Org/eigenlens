#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$PWD/work/huggingface}"
docker compose --profile qdrant up -d --wait qdrant
python -m examples.banking77_qdrant demo --query "${1:-Banking customer support problems with cards, payments, transfers, cash withdrawals and accounts}"
