"""Background job for scheduled Confluence synchronization.

Runs daily at midnight to sync Confluence pages to the RAG pipeline.
Automatically detects new and updated pages for incremental processing.
"""

import logging
import time

import schedule

from hr_chatbot_config import get_settings
from rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def sync_confluence_space() -> None:
    """Run Confluence space synchronization.

    Fetches all pages from configured Confluence space and processes
    any new or updated pages through the RAG pipeline.

    Uses incremental sync (force=False) to only process changed pages.
    Handles errors gracefully to avoid crashing the scheduled job.
    """
    try:
        logger.info("Starting scheduled Confluence sync")

        # Initialize RAG pipeline
        rag = RAGPipeline()

        # Get space key from settings
        settings = get_settings()
        space_key = settings.confluence.space_key

        logger.info(f"Syncing space: {space_key}")

        # Run sync (incremental - only new/updated pages)
        result = rag.sync_space(space_key=space_key, force=False)

        # Log completion with metrics
        logger.info(
            f"Scheduled sync completed for space '{space_key}': "
            f"{result.pages_processed} processed, "
            f"{result.pages_skipped} skipped, "
            f"{result.pages_failed} failed, "
            f"{result.chunks_created} chunks created"
        )

        # Log warning if there were failures
        if result.pages_failed > 0:
            logger.warning(
                f"{result.pages_failed} page(s) failed during sync. "
                f"Check logs for details."
            )

    except Exception as e:
        # Log error but don't re-raise
        # This allows the scheduler to continue running for next scheduled sync
        logger.error(f"Scheduled sync failed: {e}", exc_info=True)


def main() -> None:
    """Main entry point for background job.

    Schedules daily Confluence sync at midnight and runs continuously.
    Performs an initial sync on startup to ensure data is current.
    """
    logger.info("Starting Background Job Service")

    # Get and log settings
    settings = get_settings()
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Confluence space: {settings.confluence.space_key}")

    # Schedule daily sync at midnight
    logger.info("Scheduling daily sync at midnight (00:00)")
    schedule.every().day.at("00:00").do(sync_confluence_space)

    # Enter scheduling loop
    logger.info("Background job is running. Press Ctrl+C to stop.")

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    except KeyboardInterrupt:
        logger.info("Background job stopped by user")

    except Exception as e:
        logger.critical(f"Background job crashed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
