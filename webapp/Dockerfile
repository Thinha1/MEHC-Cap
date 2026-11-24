# syntax=docker/dockerfile:1.7
# One image runs both the backend (FastAPI, port 8000) and the frontend (Next.js, port 3000).
#
# Build (from webapp/, the folder of this file):
#   docker build -t klcn-webapp .
# Run (mount the model folder, it is NOT baked into the image because it is ~2.8 GB):
#   docker run --rm --gpus all -p 3000:3000 -p 8000:8000 ^
#     -v "%cd%\backend\weights:/app/webapp/backend/weights:ro" klcn-webapp
# Or use docker-compose.yml in the same folder: docker compose up --build
# Open http://localhost:3000. API docs: http://localhost:8000/docs
#
# The browser calls the backend directly, so NEXT_PUBLIC_API_BASE_URL is baked in at build time. Default
# http://localhost:8000 (same machine). To open the UI from another machine, rebuild with:
#   docker build --build-arg NEXT_PUBLIC_API_BASE_URL=http://<ip-may-chu>:8000 -t klcn-webapp .
#
# GPU: the torch cuXXX wheel ships the CUDA libraries, the host only needs the NVIDIA driver + nvidia-container-toolkit.
# Without a GPU drop --gpus all and the backend runs on CPU (slow).
# Other CUDA version: --build-arg TORCH_CUDA_TAG=cu124

ARG NODE_VERSION=22
ARG TORCH_CUDA_TAG=cu121

# ---------- 1) build frontend ----------
FROM node:${NODE_VERSION}-bookworm-slim AS frontend-build
WORKDIR /build
ENV NEXT_TELEMETRY_DISABLED=1
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}
RUN npm run build && npm prune --omit=dev

# ---------- 2) runtime image: Python (backend) + Node (frontend) ----------
FROM python:3.11-slim-bookworm AS app

ARG TORCH_CUDA_TAG

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Node is only needed to run `next start`
COPY --from=node:22-bookworm-slim /usr/local/bin/node /usr/local/bin/node

WORKDIR /app

# Python: install the CUDA build of torch first, then the rest, so pip does not pull the CPU build
# Pin transformers 4.46.1 and ultralytics 8.4.34, the versions of the Python 3.10 environment the webapp
# was tested with. transformers 5.x changes the CLIP return type (backend/pipeline/retrieval.py does not
# handle it) and newer releases need torch >= 2.6, while torch cu121 stops at 2.5.1.
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG}

COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
        "transformers==4.46.1" "ultralytics==8.4.34" \
    && python -m nltk.downloader -d /usr/share/nltk_data wordnet omw-1.4 punkt punkt_tab

# The backend must sit in the webapp.backend package (the code uses relative imports)
COPY __init__.py /app/webapp/__init__.py
COPY backend/ /app/webapp/backend/

# Prebuilt frontend
COPY --from=frontend-build /build /app/frontend

COPY <<'EOF' /app/start.sh
#!/bin/bash
# Backend in the background, frontend in the foreground. The container stops when either stops.
cd /app
python -m uvicorn webapp.backend.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!
cd /app/frontend
node node_modules/next/dist/bin/next start -H 0.0.0.0 -p 3000 &
WEB_PID=$!
trap 'kill $API_PID $WEB_PID 2>/dev/null' TERM INT
wait -n $API_PID $WEB_PID
kill $API_PID $WEB_PID 2>/dev/null
wait
EOF
RUN chmod +x /app/start.sh

ENV WEIGHTS_DIR=/app/webapp/backend/weights \
    PYTHONUNBUFFERED=1 \
    NEXT_TELEMETRY_DISABLED=1

VOLUME ["/app/webapp/backend/weights", "/app/webapp/backend/uploads"]
EXPOSE 3000 8000

# The backend loads large models at start-up, so allow a long start period
HEALTHCHECK --interval=30s --timeout=5s --start-period=240s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=4)" || exit 1

CMD ["/app/start.sh"]
