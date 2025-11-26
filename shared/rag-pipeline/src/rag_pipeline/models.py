"""Database models for Confluence page tracking.

This module defines the SQLAlchemy models for tracking synchronized Confluence pages.
The confluence_pages table enables idempotent sync operations by storing page metadata
and content hashes for change detection.
"""

import logging
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Index, Integer, JSON, String, create_engine
from sqlalchemy.orm import declarative_base

from hr_chatbot_config import get_settings

logger = logging.getLogger(__name__)

Base = declarative_base()


class ConfluencePage(Base):
    """Confluence page tracking model.

    Tracks synchronized Confluence pages for idempotency and change detection.
    Each page is uniquely identified by its Confluence page ID and stores a
    content hash to detect when pages need to be re-indexed.

    Attributes:
        id: Confluence page ID (primary key)
        title: Page title
        space_key: Confluence space key (e.g., "HR")
        content_hash: SHA256 hash of content for change detection
        last_updated: Last modified timestamp from Confluence metadata
        version: Confluence page version number
        url: Full URL to the Confluence page
        synced_at: Timestamp when we last synchronized this page
    """

    __tablename__ = "confluence_pages"

    id = Column(
        String,
        primary_key=True,
        doc="Confluence page ID"
    )
    title = Column(
        String,
        nullable=False,
        doc="Page title"
    )
    space_key = Column(
        String,
        nullable=False,
        doc="Confluence space key (e.g., 'HR')"
    )
    content_hash = Column(
        String,
        nullable=False,
        doc="SHA256 hash of content for change detection"
    )
    last_updated = Column(
        DateTime(timezone=True),
        nullable=False,
        doc="Last modified timestamp from Confluence"
    )
    version = Column(
        Integer,
        nullable=False,
        doc="Confluence page version number"
    )
    url = Column(
        String,
        nullable=False,
        doc="Full URL to the Confluence page"
    )
    synced_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        doc="When we last synchronized this page"
    )

    __table_args__ = (
        Index('idx_space_key', 'space_key'),
        Index('idx_last_updated', 'last_updated'),
    )

    def __repr__(self) -> str:
        """String representation for debugging.

        Returns:
            Formatted string with page ID and title
        """
        return f"<ConfluencePage(id={self.id}, title={self.title})>"


class ChatSession(Base):
    """Chat session model for conversation memory.

    Stores conversation history and summaries for multi-turn dialogues.
    Uses JSONB for flexible message storage with structure:
    {
        "summary": "LLM-generated conversation summary",
        "messages": ["User: ...", "Assistant: ...", ...]
    }

    Attributes:
        session_id: Unique session identifier (UUID from client)
        messages: JSONB containing summary and message array
        created_at: Session creation timestamp
        updated_at: Last update timestamp (auto-updated)
    """

    __tablename__ = "chat_sessions"

    session_id = Column(
        String,
        primary_key=True,
        doc="Unique session identifier"
    )
    messages = Column(
        JSON,
        nullable=False,
        default=lambda: {"summary": "", "messages": []},
        doc="JSONB conversation data"
    )
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        doc="Session creation time"
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        doc="Last update time"
    )

    __table_args__ = (
        Index('idx_chat_sessions_updated', 'updated_at'),
    )

    def __repr__(self) -> str:
        """String representation for debugging.

        Returns:
            Formatted string with session ID and message count
        """
        msg_count = len(self.messages.get('messages', []))
        return f"<ChatSession(session_id={self.session_id}, messages={msg_count})>"


def init_database() -> None:
    """Initialize database and create tables.

    Creates all tables defined in SQLAlchemy models using the database
    connection string from settings. This includes:
    - confluence_pages (for Confluence sync tracking)
    - chat_sessions (for conversation memory)

    This function is idempotent and safe to run multiple times.

    Raises:
        SQLAlchemyError: If database connection or table creation fails
    """
    settings = get_settings()
    logger.info("Initializing database with all tables")
    engine = create_engine(settings.database.connection_string)

    # Create all tables defined in Base
    Base.metadata.create_all(engine)

    logger.info("Database initialized successfully")

    # Log created tables
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    logger.info(f"Tables in database: {', '.join(tables)}")
