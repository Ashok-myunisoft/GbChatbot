# ─────────────────────────────────────────────
# Base image (Debian 10 → allows old TLS)
# ─────────────────────────────────────────────
FROM python:3.10-buster

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
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────
# Install Microsoft ODBC Driver 17
# ─────────────────────────────────────────────
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/debian/10/prod.list \
    > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql17

# ─────────────────────────────────────────────
# Working directory
# ─────────────────────────────────────────────
WORKDIR /app

# ─────────────────────────────────────────────
# Python dependencies
# ─────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
# Application code
# ─────────────────────────────────────────────
COPY . .

# ─────────────────────────────────────────────
# Runtime
# ─────────────────────────────────────────────
EXPOSE 8010
ENV PORT=8010

CMD ["uvicorn", "orchestrator_main:app", "--host", "0.0.0.0", "--port", "8010"]