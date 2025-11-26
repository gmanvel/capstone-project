"""Chunking service with semantic splitting and embedding generation.

This module provides document chunking using semantic boundaries rather than
fixed character counts. It generates embeddings for each chunk using
environment-aware models (Ollama for development, OpenAI for production).
"""

import logging

from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

from hr_chatbot_config import get_settings

logger = logging.getLogger(__name__)


class ChunkerService:
    """Service for semantic chunking and embedding generation.

    Splits documents into semantically coherent chunks and generates
    embeddings for each chunk. Environment-aware: uses Ollama (dev)
    or OpenAI (prod) for embeddings.
    """

    def __init__(self):
        """Initialize chunker service with environment-aware embeddings.

        Automatically selects Ollama (development) or OpenAI (production)
        based on the environment configuration. Initializes SemanticChunker
        with the configured percentile threshold.
        """
        self.settings = get_settings()

        # Initialize embeddings based on environment
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

        # Initialize semantic chunker
        self.chunker = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=self.settings.chunking.semantic_breakpoint_threshold
        )

        logger.info(
            f"ChunkerService initialized (environment: {self.settings.environment}, "
            f"threshold: {self.settings.chunking.semantic_breakpoint_threshold})"
        )

    def chunk_document(self, document: Document) -> list[Document]:
        """Split document into semantic chunks with embeddings.

        Takes a single document and splits it into semantically coherent chunks
        using SemanticChunker. Generates embeddings for all chunks in a single
        batch call for efficiency.

        Args:
            document: Document to chunk (from ConfluenceClient)

        Returns:
            List of chunk Documents with embeddings in metadata. Each chunk
            includes all original metadata plus:
            - chunk_index: 0-based position in the document
            - total_chunks: Total number of chunks from this document
            - original_page_id: Confluence page ID from original metadata
            - embedding: Embedding vector (list[float])

        Raises:
            ValueError: If document is None or has empty page_content
        """
        if document is None:
            raise ValueError("document cannot be None")

        if not document.page_content or not document.page_content.strip():
            raise ValueError("document.page_content cannot be empty")

        # Get original page ID from metadata
        original_page_id = document.metadata.get("id", "unknown")

        logger.info(f"Chunking document {original_page_id}")

        # Split text into semantic chunks
        chunk_texts = self.chunker.split_text(document.page_content)

        if not chunk_texts:
            logger.warning(f"No chunks generated for document {original_page_id}")
            return []

        total_chunks = len(chunk_texts)

        # Generate embeddings for all chunks in one batch call
        embeddings = self._generate_embeddings(chunk_texts)

        # Create chunk Documents with metadata
        chunk_documents = []
        for idx, (chunk_text, embedding) in enumerate(zip(chunk_texts, embeddings)):
            # Copy all original metadata
            chunk_metadata = document.metadata.copy()

            # Add chunk-specific metadata
            chunk_metadata["chunk_index"] = idx
            chunk_metadata["total_chunks"] = total_chunks
            chunk_metadata["original_page_id"] = original_page_id
            chunk_metadata["embedding"] = embedding

            chunk_doc = Document(
                page_content=chunk_text,
                metadata=chunk_metadata
            )
            chunk_documents.append(chunk_doc)

        logger.info(f"Chunked document {original_page_id} into {total_chunks} chunks")

        return chunk_documents

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Chunk multiple documents.

        Processes each document through chunk_document() and flattens the
        results into a single list of all chunks.

        Args:
            documents: List of documents to chunk

        Returns:
            Flattened list of all chunks from all documents

        Raises:
            ValueError: If documents is None or empty
        """
        if documents is None:
            raise ValueError("documents cannot be None")

        if not documents:
            raise ValueError("documents list cannot be empty")

        logger.info(f"Chunking {len(documents)} documents")

        all_chunks = []
        for document in documents:
            chunks = self.chunk_document(document)
            all_chunks.extend(chunks)

        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks")

        return all_chunks

    def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for text chunks.

        Calls the embedding model to generate vectors for all texts in a single
        batch call for efficiency.

        Args:
            texts: List of text chunks

        Returns:
            List of embedding vectors (each is list[float])

        Raises:
            ValueError: If texts is empty
            Exception: If embedding generation fails
        """
        if not texts:
            raise ValueError("texts list cannot be empty")

        logger.info(f"Generating embeddings for {len(texts)} chunks")

        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
            raise
