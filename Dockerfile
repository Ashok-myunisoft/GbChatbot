# ─────────────────────────────────────────────
# Base image
# ─────────────────────────────────────────────
FROM python:3.10-slim

# ─────────────────────────────────────────────
# System dependencies
#   - libpq-dev   : required by psycopg2
#   - gcc         : compile C extensions (psycopg2, faiss, etc.)
# ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev \
        gcc \
        g++ \
        build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────
# Working directory
# ─────────────────────────────────────────────
WORKDIR /app

# ─────────────────────────────────────────────
# Python dependencies (separate layer for cache)
# ─────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
# Application code
# (secrets / .env are NOT baked in — supply via
#  docker-compose env_file or runtime --env-file)
# ─────────────────────────────────────────────
COPY . .

# ─────────────────────────────────────────────
# Runtime
# ─────────────────────────────────────────────
EXPOSE 8010
ENV PORT=8010

CMD ["uvicorn", "orchestrator_main:app", "--host", "0.0.0.0", "--port", "8010"]
