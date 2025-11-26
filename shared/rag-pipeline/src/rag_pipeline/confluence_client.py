"""Confluence API client wrapper using LangChain's ConfluenceLoader.

This module provides a simplified interface for fetching Confluence pages as
LangChain Document objects with automatic content hashing for change detection.
"""

import hashlib
import logging

from langchain_community.document_loaders.confluence import ConfluenceLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ConfluenceClient:
    """Wrapper for Confluence operations using LangChain's ConfluenceLoader.

    Provides simplified interface for fetching Confluence pages as Documents.
    Uses token-based authentication (no username required).
    """

    def __init__(self, base_url: str, token: str):
        """Initialize Confluence client.

        Args:
            base_url: Confluence base URL (e.g., "https://wiki.company.com")
            token: Confluence API token

        Raises:
            ValueError: If base_url or token is empty
        """
        if not base_url or not base_url.strip():
            raise ValueError("base_url cannot be empty")

        if not token or not token.strip():
            raise ValueError("token cannot be empty")

        self.base_url = base_url.strip()
        self.token = token.strip()

        logger.info(f"Initialized Confluence client for {self.base_url}")

    def get_page_by_id(self, page_id: str) -> Document:
        """Fetch single page by ID.

        Args:
            page_id: Confluence page ID

        Returns:
            Document with page_content (plain text) and metadata.
            Metadata includes: 'id', 'title', 'source', 'when', 'content_hash'

        Raises:
            ValueError: If page_id is empty or page not found
            Exception: If ConfluenceLoader fails
        """
        if not page_id or not page_id.strip():
            raise ValueError("page_id cannot be empty")

        page_id = page_id.strip()

        logger.info(f"Fetching page {page_id}")

        loader = ConfluenceLoader(
            url=self.base_url,
            token=self.token,
            page_ids=[page_id],
            include_attachments=False,
            cloud=False
        )

        documents = loader.load()

        if not documents:
            raise ValueError(f"Page {page_id} not found")

        document = documents[0]

        # Add content_hash to metadata
        document.metadata["content_hash"] = self._compute_content_hash(document.page_content)

        title = document.metadata.get("title", "Unknown")
        logger.info(f"Fetched page {page_id}: {title}")

        return document

    def get_pages_by_space(self, space_key: str) -> list[Document]:
        """Fetch all pages from a Confluence space.

        Args:
            space_key: Confluence space key (e.g., "HR")

        Returns:
            List of Documents with page_content and metadata.
            Each metadata dict includes: 'id', 'title', 'source', 'when', 'content_hash'

        Raises:
            ValueError: If space_key is empty
            Exception: If ConfluenceLoader fails
        """
        if not space_key or not space_key.strip():
            raise ValueError("space_key cannot be empty")

        space_key = space_key.strip()

        logger.info(f"Fetching pages from space {space_key}")

        loader = ConfluenceLoader(
            url=self.base_url,
            token=self.token,
            space_key=space_key,
            include_attachments=False,
            cloud=False
        )

        # Use lazy_load() for memory efficiency
        documents = []
        for document in loader.lazy_load():
            # Add content_hash to metadata
            document.metadata["content_hash"] = self._compute_content_hash(document.page_content)
            documents.append(document)

        logger.info(f"Found {len(documents)} pages in space {space_key}")

        return documents

    @staticmethod
    def _compute_content_hash(content: str) -> str:
        """Compute SHA256 hash of content.

        Args:
            content: Text content to hash

        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
