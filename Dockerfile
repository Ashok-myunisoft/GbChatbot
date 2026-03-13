# ─────────────────────────────────────────────
# Base image
# ─────────────────────────────────────────────
# CHANGE: Upgraded from 'buster' (Debian 10, EOL) to 'bullseye' (Debian 11)
# This provides a modern, supported OS with up-to-date SSL libraries (OpenSSL 1.1.1).
# This is the primary fix for the SSL/TLS protocol error.
FROM python:3.10-bullseye

# The archive.debian.org fix is NO LONGER NEEDED and has been removed.

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
        # CHANGE: Explicitly install OpenSSL and its development headers.
        # This ensures the correct version is present for pyodbc and the MS driver.
        openssl \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────
# Install Microsoft ODBC Driver
# ─────────────────────────────────────────────
# CHANGE: Updated the Microsoft repository config from 'debian/10' to 'debian/11'
# to match the new 'bullseye' base image.
RUN curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /etc/apt/trusted.gpg.d/microsoft.gpg && \
    curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql17 mssql-tools && \
    # Clean up the ODBC driver install cache to reduce image size
    rm -rf /var/lib/apt/lists/*

# Set the path for mssql-tools (e.g., sqlcmd)
ENV PATH="$PATH:/opt/mssql-tools/bin"

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