"""RAG pipeline for document ingestion and retrieval.

This module orchestrates the complete RAG pipeline workflow: fetching documents
from Confluence, chunking them semantically, generating embeddings, and storing
vectors in Qdrant and metadata in Postgres.
"""

import logging
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from dataclasses import dataclass

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from hr_chatbot_config import get_settings

from .chunker_service import ChunkerService
from .confluence_client import ConfluenceClient
from .models import ConfluencePage

logger = logging.getLogger(__name__)

@dataclass
class SyncResult:
    pages_processed: int
    chunks_created: int
    pages_failed: int
    pages_skipped: int

class RAGPipeline:
    """RAG pipeline for document ingestion and retrieval.

    Handles Confluence content ingestion, chunking, embedding, and storage
    in both Qdrant (vectors) and Postgres (metadata). Provides idempotent
    operations with transaction safety.
    """

    def __init__(self):
        """Initialize RAG pipeline with all components.

        Sets up Confluence client, chunker service, Qdrant client, and
        database engine. Ensures Qdrant collection exists.
        """
        self.settings = get_settings()

        # Initialize Confluence client
        self.confluence_client = ConfluenceClient(
            base_url=self.settings.confluence.base_url,
            token=self.settings.confluence.token
        )

        # Initialize chunker service
        self.chunker_service = ChunkerService()

        # Initialize Qdrant client
        self.qdrant = QdrantClient(url=self.settings.qdrant.url)

        # Initialize database engine
        self.engine = create_engine(self.settings.database.connection_string)

        # Ensure Qdrant collection exists
        self._ensure_qdrant_collection()

        logger.info(f"RAGPipeline initialized (environment: {self.settings.environment})")

    def process_page(self, page_id: str, force: bool = False) -> int:
        """Process single Confluence page.

        Fetches page from Confluence, chunks it, generates embeddings,
        and stores in both Qdrant and Postgres. Idempotent: skips processing
        if page is already up-to-date (unless force=True).

        Args:
            page_id: Confluence page ID
            force: Force reprocessing even if up-to-date

        Raises:
            ValueError: If page_id is empty
            Exception: If processing fails
        """
        if not page_id or not page_id.strip():
            raise ValueError("page_id cannot be empty")

        page_id = page_id.strip()

        logger.info(f"Processing page {page_id} (force={force})")

        try:
            # Fetch document from Confluence
            document = self.confluence_client.get_page_by_id(page_id)

            with self.get_db_session() as session:
                # Check if processing needed
                if not force and not self._should_process_page(document, session):
                    logger.info(f"Page {page_id} is up-to-date, skipping")
                    return 0

                # Delete old chunks before storing new ones
                self._delete_old_chunks(page_id)

                # Chunk document
                chunks = self.chunker_service.chunk_document(document)

                if not chunks:
                    logger.warning(f"No chunks generated for page {page_id}")
                    return len(chunks)

                # Store in Qdrant
                self._store_in_qdrant(chunks)

                # Store metadata in Postgres
                self._store_in_postgres(document, session)

            logger.info(f"Successfully processed page {page_id} ({len(chunks)} chunks)")
            return len(chunks)

        except Exception as e:
            # Cleanup: delete any partial data from Qdrant
            try:
                self._delete_old_chunks(page_id)
            except Exception as cleanup_error:
                logger.error(f"Cleanup failed for page {page_id}: {cleanup_error}")

            logger.error(f"Failed to process page {page_id}: {e}", exc_info=True)
            raise 0

    def sync_space(self, space_key: str | None = None, force: bool = False) -> SyncResult:
        """Sync all pages from Confluence space.

        Processes all pages in the space, skipping up-to-date pages
        unless force=True. Continues processing even if individual
        pages fail.

        Args:
            space_key: Confluence space key (defaults to config)
            force: Force reprocessing of all pages

        Raises:
            ValueError: If space_key is empty
        """
        space_key = space_key or self.settings.confluence.space_key

        if not space_key or not space_key.strip():
            raise ValueError("space_key cannot be empty")

        space_key = space_key.strip()

        logger.info(f"Starting sync for space: {space_key} (force={force})")

        # Fetch all documents from space
        documents = self.confluence_client.get_pages_by_space(space_key)

        logger.info(f"Found {len(documents)} pages in space {space_key}")


        processed_chunks = 0
        processed_pages = 0
        skipped_pages = 0
        failed_pages = 0
        # Process each page
        for document in documents:
            page_id = document.metadata.get("id", "unknown")
            try:
                chunks = self.process_page(page_id, force=force)
                if not chunks:
                    skipped_pages += 1
                else:
                    processed_chunks += chunks
                    processed_pages += 1
            except Exception as e:
                logger.error(
                    f"Failed to process page {page_id} in space {space_key}: {e}",
                    exc_info=True
                )
                failed_pages += 1
                # Continue processing other pages

        logger.info(f"Sync completed for space {space_key}")
        return SyncResult(
            pages_processed=processed_pages,
            chunks_created=processed_chunks,
            pages_failed=failed_pages,
            pages_skipped=skipped_pages
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.7
    ) -> list[Document]:
        """Retrieve relevant documents using similarity search.

        Performs semantic search in Qdrant vector store using query embeddings.
        Returns documents with similarity scores above threshold.

        Args:
            query: User query text
            top_k: Number of results to return (1-20)
            score_threshold: Minimum similarity score (0.0-1.0)

        Returns:
            List of relevant document chunks with metadata and scores.
            Each Document includes:
            - page_content: The chunk text
            - metadata: page_id, title, chunk_index, url, score, space_key, last_modified

        Raises:
            ValueError: If query is empty or parameters are out of range
            QdrantException: If vector store query fails
            Exception: If embedding generation or retrieval fails
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k < 1 or top_k > 20:
            raise ValueError(f"top_k must be between 1 and 20, got {top_k}")

        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError(f"score_threshold must be between 0 and 1, got {score_threshold}")

        logger.info(f"Retrieving documents for query: {query[:50]}... (top_k={top_k})")

        try:
            # Generate query embedding
            query_embedding = self.chunker_service.embeddings.embed_query(query)

            # Search in Qdrant using query_points (correct API method)
            results = self.qdrant.query_points(
                collection_name=self.settings.qdrant.collection_name,
                query=query_embedding,
                limit=top_k,
                score_threshold=score_threshold
            ).points

            # Convert ScoredPoint results to Documents
            documents = []
            for result in results:
                doc = Document(
                    page_content=result.payload["text"],
                    metadata={
                        "page_id": result.payload["page_id"],
                        "title": result.payload["title"],
                        "chunk_index": result.payload["chunk_index"],
                        "url": result.payload["url"],
                        "score": result.score,
                        "space_key": result.payload.get("space_key", ""),
                        "last_modified": result.payload.get("last_modified", "")
                    }
                )
                documents.append(doc)

            logger.info(f"Retrieved {len(documents)} documents")
            return documents

        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            raise

    def _ensure_qdrant_collection(self) -> None:
        """Ensure Qdrant collection exists with correct schema.

        Creates collection if it doesn't exist, using embedding dimension
        from a test embedding. Uses cosine distance for similarity.
        """
        collections = self.qdrant.get_collections()
        collection_names = [col.name for col in collections.collections]

        collection_name = self.settings.qdrant.collection_name

        if collection_name not in collection_names:
            logger.info(f"Creating Qdrant collection: {collection_name}")

            # Generate test embedding to determine dimension
            test_embeddings = self.chunker_service._generate_embeddings(["test"])
            dimension = len(test_embeddings[0])

            logger.info(f"Using embedding dimension: {dimension}")

            # Create collection
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                )
            )

            logger.info(f"Collection created with dimension {dimension}")
        else:
            logger.info(f"Collection already exists: {collection_name}")

    def _should_process_page(self, document: Document, session: Session) -> bool:
        """Check if page needs processing.

        Compares content hash and timestamp to determine if page has changed
        since last sync.

        Args:
            document: Document from Confluence
            session: Database session

        Returns:
            True if page should be processed (new or changed), False if up-to-date
        """
        page_id = document.metadata["id"]
        content_hash = document.metadata["content_hash"]

        # Parse last_modified from ISO format
        last_modified_str = document.metadata["when"]
        last_modified = datetime.fromisoformat(last_modified_str)

        # Query existing page
        existing = session.query(ConfluencePage).filter_by(id=page_id).first()

        if not existing:
            logger.info(f"Page {page_id} is new, needs processing")
            return True

        if content_hash != existing.content_hash:
            logger.info(f"Page {page_id} content changed, needs processing")
            return True

        if last_modified > existing.last_updated:
            logger.info(f"Page {page_id} timestamp changed, needs processing")
            return True

        logger.debug(f"Page {page_id} is up-to-date")
        return False

    def _delete_old_chunks(self, page_id: str) -> None:
        """Delete old chunks from Qdrant by page_id metadata.

        Deletes all points where the payload field 'page_id' matches the given value.
        Used when reprocessing a page to avoid duplicates. Non-critical
        operation - logs warning if it fails but doesn't raise.

        Args:
            page_id: Confluence page ID to match in the 'page_id' payload field
        """
        try:
            # Delete all points where payload.page_id matches the given page_id
            self.qdrant.delete(
                collection_name=self.settings.qdrant.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="page_id",
                            match=MatchValue(value=page_id)
                        )
                    ]
                )
            )
            logger.debug(f"Deleted old chunks for page {page_id}")

        except Exception as e:
            logger.warning(f"Failed to delete old chunks for page {page_id}: {e}")

    def _store_in_qdrant(self, chunks: list[Document]) -> None:
        """Store chunk embeddings in Qdrant.

        Creates points with embeddings and metadata for semantic search.
        Uses UUID v5 to generate deterministic, unique point IDs from
        page_id and chunk_index combination.

        Args:
            chunks: List of chunk Documents with embeddings

        Raises:
            ValueError: If chunks is empty
            Exception: If Qdrant storage fails
        """
        if not chunks:
            raise ValueError("chunks list cannot be empty")

        logger.info(f"Storing {len(chunks)} chunks in Qdrant")

        points = []
        for chunk in chunks:
            # Generate UUID from page_id and chunk_index for consistent, valid IDs
            point_id_string = f"{chunk.metadata['original_page_id']}_{chunk.metadata['chunk_index']}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, point_id_string))
            
            point = PointStruct(
                id=point_id,
                vector=chunk.metadata["embedding"],
                payload={
                    "page_id": chunk.metadata["original_page_id"],
                    "title": chunk.metadata["title"],
                    "chunk_index": chunk.metadata["chunk_index"],
                    "text": chunk.page_content,
                    "space_key": chunk.metadata.get("space_key", ""),
                    "url": chunk.metadata["source"],
                    "last_modified": chunk.metadata["when"]
                }
            )
            points.append(point)

        # Upsert all points
        self.qdrant.upsert(
            collection_name=self.settings.qdrant.collection_name,
            points=points
        )

        logger.info(f"Stored {len(chunks)} chunks in Qdrant")

    def _store_in_postgres(self, document: Document, session: Session) -> None:
        """Store page metadata in Postgres.

        Creates or updates ConfluencePage record. Does not commit - relies
        on context manager.

        Args:
            document: Document from Confluence
            session: Database session
        """
        page_id = document.metadata["id"]
        title = document.metadata["title"]
        content_hash = document.metadata["content_hash"]
        url = document.metadata["source"]

        # Parse datetime from ISO format
        last_modified_str = document.metadata["when"]
        last_updated = datetime.fromisoformat(last_modified_str)

        # Extract version from metadata if available (default to 1)
        version = document.metadata.get("version", 1)

        # Extract space_key from metadata if available
        space_key = document.metadata.get("space_key", "")

        # Query existing page
        existing = session.query(ConfluencePage).filter_by(id=page_id).first()

        if existing:
            # Update existing record
            existing.title = title
            existing.content_hash = content_hash
            existing.last_updated = last_updated
            existing.version = version
            existing.url = url
            existing.synced_at = datetime.now(timezone.utc)

            logger.info(f"Updated metadata for page {page_id}")
        else:
            # Create new record
            page = ConfluencePage(
                id=page_id,
                title=title,
                space_key=space_key,
                content_hash=content_hash,
                last_updated=last_updated,
                version=version,
                url=url,
                synced_at=datetime.now(timezone.utc)
            )
            session.add(page)

            logger.info(f"Created metadata for page {page_id}")

    @contextmanager
    def get_db_session(self):
        """Database session context manager.

        Provides automatic commit on success, rollback on error,
        and ensures session is always closed.

        Yields:
            Session: SQLAlchemy session

        Example:
            with self.get_db_session() as session:
                # Do database operations
                session.query(...)
        """
        SessionLocal = sessionmaker(bind=self.engine)
        session = SessionLocal()

        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()