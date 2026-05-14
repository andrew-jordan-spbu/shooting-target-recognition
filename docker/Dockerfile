# syntax=docker/dockerfile:1.6
# ============================================================================
#  Dockerfile for the shooting_recognition Flask application.
#  Builds a self-contained image with Python 3.12, OpenCV (headless),
#  PyTorch (CPU + CUDA support in one wheel), and Ultralytics YOLOv8.
# ============================================================================
FROM python:3.12

# ── System dependencies ─────────────────────────────────────────────────────
# libgl1, libglib2.0-0  — OpenCV runtime dependencies
# ffmpeg                — video codecs (for RTSP / file sources)
# libpq-dev, gcc        — psycopg2 build dependencies (some wheels need them)
# curl                  — HEALTHCHECK
# tini                  — proper PID 1 (clean signal handling)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        libpq-dev \
        gcc \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

# ── Python environment ──────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── Dependencies (separate layer for build cache efficiency) ────────────────
# Standard PyTorch wheel contains both CPU and CUDA support — same image runs
# on hosts with and without NVIDIA GPU. Inside the app, _detect_device()
# automatically picks CUDA / MPS / CPU at runtime.
COPY docker/requirements.docker.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# ── Application sources ─────────────────────────────────────────────────────
# Build context is the repository root (see docker-compose.yml::build.context).
COPY shooting_recognition/ /app/

# Pre-create runtime directories that the app expects.
RUN mkdir -p /app/uploads /app/outputs/reports /app/outputs/detection_frames /app/recognition/weights

# ── Entrypoint ──────────────────────────────────────────────────────────────
COPY docker/scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:5000/status || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["python", "main.py"]
