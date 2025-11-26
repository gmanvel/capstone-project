"""Request and response models for API endpoints.

This module defines Pydantic models for all API request and response payloads.
Models include validation constraints and OpenAPI documentation.

Examples:
    Creating a chat request:

    >>> request = ChatRequest(
    ...     message="What is the vacation policy?",
    ...     session_id="abc123"
    ... )

    Creating a chat response:

    >>> response = ChatResponse(
    ...     response="Employees receive 15 days...",
    ...     sources=[
    ...         SourceMetadata(title="Vacation Policy", url="https://...")
    ...     ],
    ...     session_id="abc123",
    ...     confidence=0.89,
    ...     escalated=False
    ... )

    Creating a setup request:

    >>> request = SetupRequest(space_key="HR", force=False)
"""

from pydantic import BaseModel, Field

# Validation constants
MAX_MESSAGE_LENGTH = 2000
MIN_MESSAGE_LENGTH = 1
CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 1.0


# ========== CHAT ENDPOINT MODELS ==========


class ChatRequest(BaseModel):
    """Request model for chat endpoint.

    Represents a user's question to the HR chatbot with optional
    session tracking.

    Attributes:
        message: User question or message (1-2000 characters)
        session_id: Optional session ID for conversation tracking
    """

    message: str = Field(
        ...,
        min_length=MIN_MESSAGE_LENGTH,
        max_length=MAX_MESSAGE_LENGTH,
        description="User question or message",
        examples=["What is the vacation policy?"]
    )
    session_id: str | None = Field(
        None,
        description="Optional session ID for conversation tracking"
    )


class SourceMetadata(BaseModel):
    """Source document metadata.

    Contains information about a source document used to generate
    the answer.

    Attributes:
        title: Document title
        url: Link to source document
    """

    title: str = Field(..., description="Document title")
    url: str = Field(..., description="Link to source document")


class ChatResponse(BaseModel):
    """Response model for chat endpoint.

    Contains the AI-generated answer with sources and metadata.

    Attributes:
        response: AI-generated answer to the user's question
        sources: List of source documents used for the answer
        session_id: Session ID for conversation tracking
        confidence: Answer confidence score (0.0 to 1.0)
        escalated: Whether question was escalated to HR team
    """

    response: str = Field(..., description="AI-generated answer")
    sources: list[SourceMetadata] = Field(
        default_factory=list,
        description="Source documents used"
    )
    session_id: str = Field(..., description="Session ID")
    confidence: float = Field(
        ...,
        ge=CONFIDENCE_MIN,
        le=CONFIDENCE_MAX,
        description="Answer confidence score"
    )
    escalated: bool = Field(
        default=False,
        description="Whether question was escalated to HR"
    )


# ========== SETUP ENDPOINT MODELS ==========


class SetupRequest(BaseModel):
    """Request model for setup endpoint.

    Initiates RAG pipeline setup by ingesting Confluence pages from
    a specified space.

    Attributes:
        space_key: Confluence space key (uses config default if None)
        force: Force reprocessing of all pages (ignores cache)
    """

    space_key: str | None = Field(
        None,
        description="Confluence space key (defaults to config)"
    )
    force: bool = Field(
        default=False,
        description="Force reprocessing of all pages"
    )


class SetupResponse(BaseModel):
    """Response model for setup endpoint.

    Contains metrics about the setup process including pages processed,
    skipped, and failed.

    Attributes:
        status: Setup status: "completed", "failed", or "in_progress"
        space_key: Confluence space key that was processed
        pages_processed: Number of pages successfully processed
        pages_skipped: Number of pages skipped (unchanged)
        pages_failed: Number of pages that failed processing
        duration_seconds: Total processing time in seconds
        chunks_created: Total number of document chunks created
    """

    status: str = Field(..., description="Setup status: completed, failed, in_progress")
    space_key: str = Field(..., description="Processed space key")
    pages_processed: int = Field(..., description="Number of pages processed")
    pages_skipped: int = Field(..., description="Number of pages skipped")
    pages_failed: int = Field(..., description="Number of pages that failed")
    duration_seconds: float = Field(..., description="Total processing time")
    chunks_created: int = Field(..., description="Total chunks created")
