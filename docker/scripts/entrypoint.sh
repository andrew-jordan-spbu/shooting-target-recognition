#!/usr/bin/env bash
# ============================================================================
#  entrypoint.sh — container startup script.
#  1. Wait for PostgreSQL to accept connections (belt-and-braces on top of
#     Compose's depends_on healthcheck).
#  2. Report detected compute device (CUDA / CPU; MPS is not exposed to Docker).
#  3. Warn if best.pt is missing — app will fall back to mock mode.
#  4. Exec the CMD (python main.py) — keeps PID 1 == app process via tini.
# ============================================================================
set -euo pipefail

DB_HOST="${DB_HOST:-db}"
DB_PORT="${DB_PORT:-5432}"
DB_USER="${DB_USER:-postgres}"
DB_NAME="${DB_NAME:-shooting_recognition}"
WEIGHTS=/app/recognition/weights/best.pt

# ── 1. Wait for Postgres ────────────────────────────────────────────────────
echo "[entrypoint] Waiting for PostgreSQL at ${DB_HOST}:${DB_PORT} ..."
for i in $(seq 1 60); do
    if python - <<PY 2>/dev/null
import os, sys, psycopg2
try:
    psycopg2.connect(
        host=os.environ.get("DB_HOST", "db"),
        port=int(os.environ.get("DB_PORT", 5432)),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", ""),
        dbname=os.environ.get("DB_NAME", "shooting_recognition"),
        connect_timeout=2,
    ).close()
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
    then
        echo "[entrypoint] ✓ PostgreSQL is reachable (attempt $i)"
        break
    fi
    if [ "$i" = 60 ]; then
        echo "[entrypoint] ✗ PostgreSQL still unreachable after 60 attempts — aborting"
        exit 1
    fi
    sleep 1
done

# ── 2. Detect compute device ────────────────────────────────────────────────
python - <<'PY'
import os
try:
    import torch
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        print(f"[entrypoint] ✓ CUDA available: {n}× {name}")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # Note: this branch is effectively unreachable inside Docker on macOS;
        # included for completeness if someone runs this image natively.
        print("[entrypoint] ✓ Apple MPS available")
    else:
        print("[entrypoint] ℹ  No GPU detected — using CPU (inference will be slower)")
except Exception as exc:
    print(f"[entrypoint] ⚠  Could not query device: {exc}")
PY

# ── 3. Check model weights ──────────────────────────────────────────────────
if [ -f "$WEIGHTS" ]; then
    size_mb=$(du -m "$WEIGHTS" | cut -f1)
    echo "[entrypoint] ✓ Model weights present at $WEIGHTS (${size_mb} MB)"
else
    echo "[entrypoint] ⚠  $WEIGHTS not found — app will start in MOCK mode."
    echo "[entrypoint]    Mount best.pt to that path via docker-compose volume."
fi

# ── 4. Hand off to the application ──────────────────────────────────────────
echo "[entrypoint] → exec: $*"
exec "$@"
