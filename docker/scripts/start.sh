#!/usr/bin/env bash
# ============================================================================
#  start.sh — wrapper around `docker compose` that auto-detects NVIDIA GPUs
#  and conditionally includes the GPU-override file.
#
#  Usage (from docker/ directory):
#      ./scripts/start.sh up --build
#      ./scripts/start.sh up -d
#      ./scripts/start.sh down
#      ./scripts/start.sh logs -f app
#
#  Pass any docker compose subcommand and flags; they're forwarded unchanged.
# ============================================================================
set -euo pipefail

# Locate compose files relative to this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COMPOSE_DIR="$( dirname "$SCRIPT_DIR" )"

cd "$COMPOSE_DIR"

# Decide whether to layer in the GPU override file.
GPU_FLAGS=()
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "✓ NVIDIA GPU detected — including GPU override (docker-compose.gpu.yml)"
    GPU_FLAGS+=(-f docker-compose.yml -f docker-compose.gpu.yml)
else
    echo "ℹ  No NVIDIA GPU detected — CPU inference mode"
    GPU_FLAGS+=(-f docker-compose.yml)
fi

# Pick docker compose v2 (preferred) or fall back to legacy docker-compose.
if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD=(docker-compose)
else
    echo "[ERROR] Neither 'docker compose' nor 'docker-compose' is installed."
    exit 1
fi

echo "→ ${COMPOSE_CMD[*]} ${GPU_FLAGS[*]} $*"
exec "${COMPOSE_CMD[@]}" "${GPU_FLAGS[@]}" "$@"
