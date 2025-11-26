"""Setup endpoint router for initial RAG pipeline setup.

This module provides the /setup endpoint that performs one-time ingestion
of Confluence pages into the RAG pipeline.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, status

from hr_chatbot_config import get_settings
from rag_pipeline import RAGPipeline, SyncResult

from ..models import SetupRequest, SetupResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/setup", response_model=SetupResponse, status_code=status.HTTP_200_OK)
def setup(request: SetupRequest) -> SetupResponse:
    """Initial RAG pipeline setup - ingest all pages from Confluence space.

    This endpoint performs initial ingestion of Confluence pages. Should typically
    be run once after deployment. Can be re-run with force=True to reprocess all
    pages regardless of their current state.

    Workflow:
    1. Fetch all pages from Confluence space
    2. For each page: check content hash, chunk if needed, embed, store
    3. Skip pages that are already up-to-date (unless force=True)
    4. Store chunks in Qdrant (vectors) and page metadata in Postgres

    Args:
        request: Setup request with optional space_key and force flag

    Returns:
        SetupResponse with status, duration, and space key

    Raises:
        HTTPException 400: Invalid space key or parameters
        HTTPException 500: Setup processing failed

    Note:
        Detailed metrics (pages_processed, pages_skipped, etc.) are set to 0
        because RAG pipeline sync_space() is a void method. The important
        metrics are status ("completed") and duration_seconds.
    """
    logger.info(
        f"Setup request received: space_key={request.space_key}, force={request.force}"
    )

    start_time = datetime.utcnow()

    try:
        # Initialize RAG pipeline
        rag = RAGPipeline()

        # Get space key (from request or config default)
        settings = get_settings()
        space_key = request.space_key or settings.confluence.space_key

        # Sync space (void method - returns None)
        # This fetches all pages, chunks them, and stores in Qdrant + Postgres
        sync_result: SyncResult = rag.sync_space(space_key=space_key, force=request.force)

        # Calculate duration
        duration = (datetime.utcnow() - start_time).total_seconds()

        # Build response
        # Note: sync_space() is void, so detailed metrics are unavailable
        response = SetupResponse(
            status="completed",
            space_key=space_key,
            pages_processed=sync_result.pages_processed,
            pages_skipped=sync_result.pages_skipped,
            pages_failed=sync_result.pages_failed,
            duration_seconds=duration,
            chunks_created=sync_result.chunks_created,
        )

        logger.info(f"Setup completed in {duration:.1f}s for space '{space_key}'")

        return response

    except ValueError as e:
        # Invalid space key or parameters
        logger.warning(f"Invalid setup request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    except Exception as e:
        # Setup failed
        logger.error(f"Setup failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Setup failed: {str(e)}"
        )
