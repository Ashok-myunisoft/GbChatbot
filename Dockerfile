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
        gcc \
        g++ \
        build-essential \
    && \
    # ── Microsoft MSSQL ODBC 17 driver ───────────────────────────────────
    # Download and register the Microsoft signing key
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg && \
    # Manually write the sources entry with explicit signed-by so apt trusts it
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/11/prod bullseye main" \
        > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql17 && \
    # ── Allow TLS 1.0/1.1 for older on-premise MSSQL servers ─────────────
    # Debian 11 OpenSSL enforces TLSv1.2 minimum by default.
    # Older MSSQL servers only support TLS 1.0 — lower the floor so the
    # SSL handshake succeeds (unsupported protocol error fix).
    sed -i 's/MinProtocol = TLSv1.2/MinProtocol = TLSv1/g' /etc/ssl/openssl.cnf && \
    sed -i 's/CipherString = DEFAULT@SECLEVEL=2/CipherString = DEFAULT@SECLEVEL=1/g' /etc/ssl/openssl.cnf && \
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
