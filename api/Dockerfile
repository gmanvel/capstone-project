# Multi-stage build for API service
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
RUN uv sync && uv pip install -e ./api -e ./shared/config -e ./shared/rag-pipeline

# Production stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --from=builder /app/api /app/api
COPY --from=builder /app/shared /app/shared

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH" \
    PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run uvicorn server
CMD ["python", "-m", "uvicorn", "hr_chatbot_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
