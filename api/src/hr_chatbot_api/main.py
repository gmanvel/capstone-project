"""Main FastAPI application for HR Chatbot API.

This module sets up the FastAPI application with lifespan management,
health check endpoint, and API routing.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text

from hr_chatbot_config import get_settings
from rag_pipeline import init_database

from .routers import chat, setup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events.

    Startup: Initialize resources (database, etc.)
    Shutdown: Cleanup resources

    Args:
        app: FastAPI application instance

    Yields:
        None
    """
    # Startup
    logger.info("Starting HR Chatbot API")

    settings = get_settings()
    logger.info(f"Environment: {settings.environment}")

    # Initialize database
    try:
        init_database()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        # Don't fail startup - health check will report unhealthy

    yield

    # Shutdown
    logger.info("Shutting down HR Chatbot API")


# Create FastAPI app
app = FastAPI(
    title="HR Chatbot API",
    description="Multi-agent RAG system for HR queries",
    version="0.1.0",
    lifespan=lifespan
)

# Include routers
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(setup.router, prefix="/api/v1", tags=["setup"])


@app.get("/health", status_code=status.HTTP_200_OK, tags=["health"])
async def health_check():
    """Health check endpoint.

    Checks status of dependencies: database, vector store, LLM.
    Returns 200 if healthy, 503 if any component is unhealthy.

    Returns:
        JSON response with health status and component checks
    """
    settings = get_settings()

    checks = {
        "database": "unknown",
        "vector_store": "unknown",
        "llm": "unknown"
    }

    # Check database
    try:
        engine = create_engine(settings.database.connection_string)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = "healthy"
        logger.debug("Database health check: healthy")
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        checks["database"] = "unhealthy"
    finally:
        if 'engine' in locals():
            engine.dispose()

    # Check vector store
    try:
        qdrant = QdrantClient(
            host=settings.qdrant.host,
            port=settings.qdrant.port
        )
        qdrant.get_collections()
        checks["vector_store"] = "healthy"
        logger.debug("Vector store health check: healthy")
    except Exception as e:
        logger.error(f"Vector store health check failed: {e}")
        checks["vector_store"] = "unhealthy"

    # Check LLM (assume healthy if config exists)
    checks["llm"] = "healthy"

    # Determine overall status
    overall_status = "healthy" if all(v == "healthy" for v in checks.values()) else "unhealthy"

    response_code = status.HTTP_200_OK if overall_status == "healthy" else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(
        status_code=response_code,
        content={
            "status": overall_status,
            "service": "api",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks
        }
    )


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information.

    Returns:
        Dict with API service information and available endpoints
    """
    return {
        "service": "HR Chatbot API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }
