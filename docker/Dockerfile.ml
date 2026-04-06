# docker/Dockerfile.ml
# VIT Sports Intelligence Network - ML Service Container

FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    libopenblas-dev \
    liblapack-dev \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r vituser && \
    useradd -r -g vituser -u 1000 -m -s /bin/bash vituser && \
    chown -R vituser:vituser /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=vituser:vituser services/ml_service/ ./services/ml_service/
COPY --chown=vituser:vituser app/ ./app/
COPY --chown=vituser:vituser main.py .
COPY --chown=vituser:vituser alembic.ini .
COPY --chown=vituser:vituser alembic/ ./alembic/

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs && \
    chown -R vituser:vituser /app/models /app/data /app/logs

# Switch to non-root user
USER vituser

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the ML service
CMD ["uvicorn", "services.ml_service.main:app", "--host", "0.0.0.0", "--port", "8001"]