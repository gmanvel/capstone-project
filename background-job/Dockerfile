# Multi-stage build for Background Job service
FROM python:3.11-slim AS builder

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy workspace configuration
COPY pyproject.toml uv.lock ./

# Copy shared dependencies
COPY shared/config ./shared/config
COPY shared/rag-pipeline ./shared/rag-pipeline

# Copy all services (needed for workspace resolution)
COPY api ./api
COPY background-job ./background-job

# Install workspace and all member packages
RUN uv sync && uv pip install -e ./background-job -e ./shared/config -e ./shared/rag-pipeline

# Production stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --from=builder /app/background-job /app/background-job
COPY --from=builder /app/shared /app/shared

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH" \
    PYTHONUNBUFFERED=1

# Run scheduled sync script
CMD ["python", "-m", "sync.main"]
