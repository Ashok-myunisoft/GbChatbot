# ─────────────────────────────────────────────
# Base image
# ─────────────────────────────────────────────
FROM python:3.10-slim

# ─────────────────────────────────────────────
# System dependencies
#   - curl, gnupg, apt-transport-https : needed to add Microsoft ODBC repo
#   - unixodbc-dev                     : required by pyodbc
#   - freetds-dev, freetds-bin         : required by pymssql
#   - gcc, g++, build-essential        : compile C extensions (pymssql, faiss, etc.)
# ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        apt-transport-https \
        ca-certificates \
        unixodbc \
        unixodbc-dev \
        freetds-dev \
        freetds-bin \
        tdsodbc \
        gcc \
        g++ \
        build-essential \
    && \
    # ── Microsoft MSSQL ODBC 18 driver ───────────────────────────────────
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg && \
    curl -fsSL https://packages.microsoft.com/config/debian/11/prod.list \
        -o /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql17 && \
    # ── Cleanup ──────────────────────────────────────────────────────────
    apt-get clean && rm -rf /var/lib/apt/lists/*

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
