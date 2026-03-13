# ─────────────────────────────────────────────
# Base image - Upgraded to Debian 12 (Bookworm)
# ─────────────────────────────────────────────
FROM python:3.11-bookworm

# ─────────────────────────────────────────────
# System dependencies
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
        openssl \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────
# Install Microsoft ODBC Driver 18
# ─────────────────────────────────────────────
# Updated to use the Debian 12 repo and msodbcsql18
RUN curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /etc/apt/trusted.gpg.d/microsoft.gpg && \
    curl https://packages.microsoft.com/config/debian/12/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 mssql-tools18 && \
    rm -rf /var/lib/apt/lists/*

# Set the path for mssql-tools18
ENV PATH="$PATH:/opt/mssql-tools18/bin"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8010
ENV PORT=8010

CMD ["uvicorn", "orchestrator_main:app", "--host", "0.0.0.0", "--port", "8010"]