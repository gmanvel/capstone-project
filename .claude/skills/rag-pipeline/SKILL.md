# RAG Pipeline Implementation Patterns

## Overview

This skill provides patterns for implementing the RAG (Retrieval-Augmented Generation) pipeline that handles:
- Confluence content ingestion
- Text chunking
- Embedding generation (environment-aware)
- Vector storage in Qdrant
- Metadata storage in Postgres
- Semantic retrieval

## Core Principles

1. **Idempotency**: All operations are safe to retry (hash-based change detection)
2. **Environment Parity**: Dev (Ollama) and Prod (OpenAI) behavior must match
3. **Transaction Safety**: All-or-nothing updates (rollback on failure)
4. **Metadata Tracking**: Store enough metadata for debugging and source attribution

## RAGPipeline Class Structure

```python
from typing import Any
import logging
from datetime import datetime
import hashlib

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from hr_chatbot_config import get_settings

logger = logging.getLogger(__name__)

class RAGPipeline:
    """RAG pipeline for document ingestion and retrieval.
    
    Handles Confluence content ingestion, chunking, embedding, and storage.
    Environment-aware: automatically uses Ollama (dev) or OpenAI (prod).
    """
    
    def __init__(self):
        """Initialize RAG pipeline with configuration."""
        self.settings = get_settings()
        
        # Initialize components
        self._init_embeddings()
        self._init_text_splitter()
        self._init_qdrant()
        self._init_database()
        
        logger.info(f"RAG pipeline initialized (environment: {self.settings.environment})")
    
    def _init_embeddings(self):
        """Initialize embeddings based on environment."""
        if self.settings.llm.is_local:
            logger.info("Using Ollama embeddings for development")
            self.embeddings = OllamaEmbeddings(
                base_url=self.settings.llm.ollama_base_url,
                model=self.settings.llm.embedding_model_name
            )
        else:
            logger.info("Using OpenAI embeddings for production")
            self.embeddings = OpenAIEmbeddings(
                api_key=self.settings.llm.openai_api_key,
                model=self.settings.llm.embedding_model_name
            )
    
    def _init_text_splitter(self):
        """Initialize text splitter with chunking parameters."""
        chunker_kwargs["breakpoint_threshold_amount"] = (
                    self.config.semantic_breakpoint_threshold
                )
        self.text_splitter = SemanticChunker(**chunker_kwargs)
    
    def _init_qdrant(self):
        """Initialize Qdrant client and ensure collection exists."""
        self.qdrant = QdrantClient(
            host=self.settings.qdrant.host,
            port=self.settings.qdrant.port
        )
        self._ensure_collection()
    
    def _init_database(self):
        """Initialize database engine."""
        self.engine = create_engine(self.settings.database.connection_string)
```

## Qdrant Collection Setup

```python
def _ensure_collection(self):
    """Ensure Qdrant collection exists with correct schema.
    
    Creates collection if it doesn't exist. Collection configuration
    matches embedding dimension (768 for Ollama, 1536 for OpenAI).
    """
    collections = self.qdrant.get_collections()
    collection_names = [col.name for col in collections.collections]
    
    if self.settings.qdrant.collection_name not in collection_names:
        logger.info(f"Creating Qdrant collection: {self.settings.qdrant.collection_name}")
        
        # Get embedding dimension from test embedding
        test_embedding = self.embeddings.embed_query("test")
        dimension = len(test_embedding)
        
        logger.info(f"Using embedding dimension: {dimension}")
        
        self.qdrant.create_collection(
            collection_name=self.settings.qdrant.collection_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE
            )
        )
        logger.info("Collection created successfully")
    else:
        logger.info(f"Collection already exists: {self.settings.qdrant.collection_name}")
```

## Idempotency Pattern

```python
def should_process_page(
    self,
    session: Session,
    page_id: str,
    confluence_last_modified: datetime,
    content_hash: str
) -> bool:
    """Check if page needs processing.
    
    Uses both content hash and timestamp to detect changes.
    
    Args:
        session: Database session
        page_id: Confluence page ID
        confluence_last_modified: Last modified timestamp from Confluence
        content_hash: SHA256 hash of current content
        
    Returns:
        True if page should be processed (new or changed), False if up-to-date
    """
    from .models import ConfluencePage
    
    existing = session.query(ConfluencePage).filter_by(id=page_id).first()
    
    if not existing:
        logger.info(f"Page {page_id} is new, needs processing")
        return True
    
    if content_hash != existing.content_hash:
        logger.info(f"Page {page_id} content changed, needs processing")
        return True
    
    if confluence_last_modified > existing.last_updated:
        logger.info(f"Page {page_id} timestamp changed, needs processing")
        return True
    
    logger.debug(f"Page {page_id} is up-to-date, skipping")
    return False


@staticmethod
def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content for change detection.
    
    Args:
        content: Text content to hash
        
    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()
```

## Text Chunking Pattern

```python
def chunk_text(self, text: str) -> list[str]:
    """Split text into overlapping chunks.
    
    Uses RecursiveCharacterTextSplitter with:
    - Chunk size: 1000 characters
    - Overlap: 200 characters
    - Separators: paragraph, line, sentence, word
    
    Args:
        text: Text to chunk
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for chunking")
        return []
    
    chunks = self.text_splitter.split_text(text)
    logger.info(f"Split text into {len(chunks)} chunks")
    
    return chunks
```

## Embedding Generation Pattern

```python
def generate_embeddings(self, chunks: list[str]) -> list[list[float]]:
    """Generate embeddings for text chunks.
    
    Automatically uses Ollama (dev) or OpenAI (prod) based on environment.
    
    Args:
        chunks: List of text chunks
        
    Returns:
        List of embedding vectors
        
    Raises:
        ValueError: If chunks is empty
        Exception: If embedding generation fails
    """
    if not chunks:
        raise ValueError("Cannot generate embeddings for empty chunks list")
    
    logger.info(f"Generating embeddings for {len(chunks)} chunks")
    
    try:
        embeddings = self.embeddings.embed_documents(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
        raise
```

## Qdrant Storage Pattern

```python
def store_in_qdrant(
    self,
    page_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
    metadata: dict[str, Any]
) -> None:
    """Store embeddings in Qdrant with metadata.
    
    Creates points with format: {page_id}_{chunk_index}
    Includes full text and metadata in payload for retrieval.
    
    Args:
        page_id: Confluence page ID
        chunks: List of text chunks
        embeddings: List of embedding vectors
        metadata: Page metadata (title, url, space_key, last_modified)
        
    Raises:
        ValueError: If chunks and embeddings length mismatch
        QdrantException: If storage fails
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunks and embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
        )
    
    logger.info(f"Storing {len(chunks)} chunks in Qdrant for page {page_id}")
    
    points = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point = PointStruct(
            id=f"{page_id}_{idx}",
            vector=embedding,
            payload={
                "page_id": page_id,
                "title": metadata["title"],
                "chunk_index": idx,
                "text": chunk,
                "space_key": metadata["space_key"],
                "url": metadata["url"],
                "last_modified": metadata["last_modified"]
            }
        )
        points.append(point)
    
    try:
        self.qdrant.upsert(
            collection_name=self.settings.qdrant.collection_name,
            points=points
        )
        logger.info(f"Successfully stored {len(points)} points in Qdrant")
    
    except Exception as e:
        logger.error(f"Failed to store in Qdrant: {e}", exc_info=True)
        raise
```

## Postgres Metadata Storage Pattern

```python
def store_metadata(
    self,
    session: Session,
    page_id: str,
    title: str,
    space_key: str,
    content_hash: str,
    last_updated: datetime,
    version: int,
    url: str
) -> None:
    """Store or update page metadata in Postgres.
    
    Args:
        session: Database session
        page_id: Confluence page ID
        title: Page title
        space_key: Confluence space key
        content_hash: SHA256 hash of content
        last_updated: Last modified timestamp
        version: Page version number
        url: Full page URL
    """
    from .models import ConfluencePage
    
    existing = session.query(ConfluencePage).filter_by(id=page_id).first()
    
    if existing:
        # Update existing record
        logger.info(f"Updating metadata for page {page_id}")
        existing.title = title
        existing.space_key = space_key
        existing.content_hash = content_hash
        existing.last_updated = last_updated
        existing.version = version
        existing.url = url
        existing.synced_at = datetime.utcnow()
    else:
        # Create new record
        logger.info(f"Creating metadata for page {page_id}")
        page = ConfluencePage(
            id=page_id,
            title=title,
            space_key=space_key,
            content_hash=content_hash,
            last_updated=last_updated,
            version=version,
            url=url,
            synced_at=datetime.utcnow()
        )
        session.add(page)
```

## Transaction-Safe Processing Pattern

```python
def process_page(self, page_id: str, force: bool = False) -> dict[str, Any]:
    """Process a single Confluence page with transaction safety.
    
    Steps:
    1. Fetch content from Confluence
    2. Check if processing needed (unless force=True)
    3. Chunk text
    4. Generate embeddings
    5. Store in Qdrant
    6. Store metadata in Postgres
    7. Commit transaction
    
    On failure: Rollback DB and cleanup Qdrant.
    
    Args:
        page_id: Confluence page ID
        force: Force reprocessing even if up-to-date
        
    Returns:
        Processing result with status and metrics
        
    Raises:
        ValueError: If page_id is invalid
        Exception: If processing fails
    """
    if not page_id or not page_id.strip():
        raise ValueError("page_id cannot be empty")
    
    logger.info(f"Processing page {page_id} (force={force})")
    
    from contextlib import contextmanager
    from sqlalchemy.orm import sessionmaker
    
    @contextmanager
    def get_session():
        Session = sessionmaker(bind=self.engine)
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    with get_session() as session:
        try:
            # 1. Fetch from Confluence
            confluence_data = self._fetch_page(page_id)
            text = self._extract_text(confluence_data["body"]["storage"]["value"])
            content_hash = self.compute_content_hash(text)
            
            # 2. Check if processing needed
            if not force:
                needs_processing = self.should_process_page(
                    session=session,
                    page_id=page_id,
                    confluence_last_modified=datetime.fromisoformat(
                        confluence_data["version"]["when"].replace('Z', '+00:00')
                    ),
                    content_hash=content_hash
                )
                
                if not needs_processing:
                    return {
                        "status": "skipped",
                        "page_id": page_id,
                        "reason": "already up-to-date"
                    }
            
            # 3. Delete old embeddings if updating
            self._delete_old_embeddings(page_id)
            
            # 4. Chunk text
            chunks = self.chunk_text(text)
            if not chunks:
                logger.warning(f"No chunks generated for page {page_id}")
                return {
                    "status": "skipped",
                    "page_id": page_id,
                    "reason": "no content to process"
                }
            
            # 5. Generate embeddings
            embeddings = self.generate_embeddings(chunks)
            
            # 6. Store in Qdrant
            metadata = {
                "title": confluence_data["title"],
                "space_key": confluence_data["space"]["key"],
                "url": self._build_page_url(page_id),
                "last_modified": confluence_data["version"]["when"]
            }
            self.store_in_qdrant(page_id, chunks, embeddings, metadata)
            
            # 7. Store metadata in Postgres
            self.store_metadata(
                session=session,
                page_id=page_id,
                title=confluence_data["title"],
                space_key=confluence_data["space"]["key"],
                content_hash=content_hash,
                last_updated=datetime.fromisoformat(
                    confluence_data["version"]["when"].replace('Z', '+00:00')
                ),
                version=confluence_data["version"]["number"],
                url=self._build_page_url(page_id)
            )
            
            logger.info(f"Successfully processed page {page_id} ({len(chunks)} chunks)")
            
            return {
                "status": "processed",
                "page_id": page_id,
                "chunks_created": len(chunks),
                "title": confluence_data["title"]
            }
        
        except Exception as e:
            logger.error(f"Failed to process page {page_id}: {e}", exc_info=True)
            
            # Cleanup: Delete partial Qdrant data
            try:
                self._delete_old_embeddings(page_id)
            except Exception as cleanup_error:
                logger.error(f"Cleanup failed: {cleanup_error}")
            
            raise


def _delete_old_embeddings(self, page_id: str) -> None:
    """Delete old embeddings for a page from Qdrant.
    
    Used when reprocessing a page to avoid duplicates.
    
    Args:
        page_id: Confluence page ID
    """
    try:
        self.qdrant.delete(
            collection_name=self.settings.qdrant.collection_name,
            points_selector={
                "filter": {
                    "must": [
                        {"key": "page_id", "match": {"value": page_id}}
                    ]
                }
            }
        )
        logger.debug(f"Deleted old embeddings for page {page_id}")
    except Exception as e:
        logger.warning(f"Failed to delete old embeddings: {e}")
```

## Retrieval Pattern

```python
def retrieve(
    self,
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.7
) -> list[dict[str, Any]]:
    """Retrieve relevant documents using similarity search.
    
    Args:
        query: User query text
        top_k: Number of results to return (1-20)
        score_threshold: Minimum similarity score (0.0-1.0)
        
    Returns:
        List of document chunks with metadata and scores
        
    Raises:
        ValueError: If parameters are invalid
        QdrantException: If search fails
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
        query_embedding = self.embeddings.embed_query(query)
        
        # Search in Qdrant
        results = self.qdrant.search(
            collection_name=self.settings.qdrant.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=score_threshold
        )
        
        # Format results
        documents = []
        for result in results:
            documents.append({
                "text": result.payload["text"],
                "page_id": result.payload["page_id"],
                "title": result.payload["title"],
                "chunk_index": result.payload["chunk_index"],
                "url": result.payload["url"],
                "score": result.score,
                "space_key": result.payload["space_key"]
            })
        
        logger.info(f"Retrieved {len(documents)} documents")
        return documents
    
    except Exception as e:
        logger.error(f"Retrieval failed: {e}", exc_info=True)
        raise
```

## Batch Processing Pattern

```python
def sync_space(
    self,
    space_key: str | None = None,
    force: bool = False
) -> dict[str, Any]:
    """Sync all pages from a Confluence space.
    
    Args:
        space_key: Space key to sync (defaults to config)
        force: Force reprocessing of all pages
        
    Returns:
        Sync summary with metrics
    """
    space_key = space_key or self.settings.confluence.space_key
    logger.info(f"Starting sync for space: {space_key} (force={force})")
    
    start_time = datetime.utcnow()
    
    # Fetch all pages
    pages = self._fetch_space_pages(space_key)
    logger.info(f"Found {len(pages)} pages in space {space_key}")
    
    results = {
        "total_pages": len(pages),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "failed_pages": []
    }
    
    # Process each page
    for page in pages:
        try:
            result = self.process_page(page["id"], force=force)
            
            if result["status"] == "processed":
                results["processed"] += 1
            elif result["status"] == "skipped":
                results["skipped"] += 1
        
        except Exception as e:
            logger.error(f"Failed to process page {page['id']}: {e}")
            results["failed"] += 1
            results["failed_pages"].append({
                "page_id": page["id"],
                "title": page.get("title", "Unknown"),
                "error": str(e)
            })
    
    duration = (datetime.utcnow() - start_time).total_seconds()
    results["duration_seconds"] = duration
    
    logger.info(
        f"Sync completed: {results['processed']} processed, "
        f"{results['skipped']} skipped, {results['failed']} failed "
        f"({duration:.1f}s)"
    )
    
    return results
```

## Key Takeaways

1. **Always use environment-aware embeddings** - Check `settings.llm.is_local`
2. **Implement idempotency** - Hash-based change detection prevents duplicate work
3. **Use transactions** - Rollback DB and cleanup Qdrant on failure
4. **Validate inputs early** - Fail fast with clear error messages
5. **Log comprehensively** - Include context (page_id, chunk count, duration)
6. **Store rich metadata** - Include everything needed for debugging and source attribution
7. **Handle partial failures gracefully** - One page failure shouldn't stop entire sync
8. **Use appropriate chunk sizes** - 1000 chars with 200 overlap is a good default