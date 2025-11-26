# Memory Service Patterns

## Overview

This skill provides patterns for implementing session-based conversation memory in the HR RAG chatbot. Conversation memory enables multi-turn dialogues by storing conversation history and automatically summarizing older messages to stay within token limits.

**Purpose**: Enable context-aware responses across multiple conversation turns

**Scope**: Chat session management, message persistence, threshold-based summarization

**Technology**:
- **Storage**: PostgreSQL with JSONB column
- **ORM**: SQLAlchemy 2.0+
- **Summarization**: LangChain with environment-aware LLM (Ollama/OpenAI)
- **Configuration**: Pydantic Settings

## Core Principles

### 1. Lightweight Design
Store only what's necessary for context. Use a single JSONB column to avoid schema complexity.

```python
# Good: Simple JSONB structure
{
    "summary": "User asked about vacation policy. Provided 15 days info.",
    "messages": [
        "User: Can I carry over days?",
        "Assistant: Yes, up to 5 days can be carried over."
    ]
}

# Bad: Over-engineered schema with multiple tables
# - conversations table
# - messages table (one row per message)
# - summaries table
# - message_embeddings table
```

### 2. Threshold-Based Summarization
Don't summarize every turn. Wait until a threshold is reached (default: 6 messages).

**Why threshold-based?**
- Reduces LLM API calls (cost savings)
- Better context retention (recent messages preserved verbatim)
- Predictable behavior (always triggers at N messages)

### 3. Idempotent Operations
All memory operations must be safe to retry without side effects.

```python
# Good: Check before insert
def get_session_memory(session_id: str) -> dict[str, Any]:
    session = query(ChatSession).filter_by(session_id=session_id).first()
    if not session:
        return {"summary": "", "messages": []}  # Default, never fail
    return session.messages

# Bad: Assume session exists
def get_session_memory(session_id: str) -> dict[str, Any]:
    session = query(ChatSession).filter_by(session_id=session_id).first()
    return session.messages  # Crashes if session is None
```

### 4. Session-Scoped Isolation
Each session's memory is completely isolated. No cross-session access.

```python
# Good: Session ID is always required and validated
def append_message(session_id: str, role: str, content: str) -> None:
    if not session_id or not session_id.strip():
        raise ValueError("session_id is required")
    # Process for this session only

# Bad: Global memory or user-scoped memory
def append_message(user_id: str, role: str, content: str) -> None:
    # Mixing concerns - memory should be session-scoped, not user-scoped
    pass
```

## Data Model Pattern

### SQLAlchemy Model Definition

```python
"""Database models for conversation memory."""

from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Index, JSON, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


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
        """String representation for debugging."""
        msg_count = len(self.messages.get('messages', []))
        return f"<ChatSession(session_id={self.session_id}, messages={msg_count})>"
```

**Key Points**:
- **JSON column type**: SQLAlchemy automatically uses JSONB on PostgreSQL
- **Lambda defaults**: Prevents mutable default argument issues
- **Timezone-aware timestamps**: Always use `DateTime(timezone=True)`
- **onupdate lambda**: Automatically updates `updated_at` on modifications
- **Index on updated_at**: Enables efficient "recent sessions" queries

## CRUD Patterns

### Initialize MemoryService

```python
"""Memory service for conversation management."""

import logging
from typing import Any
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

from hr_chatbot_config import get_settings
from rag_pipeline.models import ChatSession

logger = logging.getLogger(__name__)


class MemoryService:
    """Service for managing conversation memory.

    Handles session storage, message persistence, and LLM-based summarization
    of conversation history. Uses threshold-based summarization to keep memory
    size manageable while preserving context.
    """

    def __init__(self):
        """Initialize memory service with settings and LLM."""
        self.settings = get_settings()
        self.engine = create_engine(self.settings.database.connection_string)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self._init_llm()

    def _init_llm(self):
        """Initialize environment-aware LLM for summarization."""
        if self.settings.llm.is_local:
            # Development: Ollama
            from langchain_community.chat_models import ChatOllama

            self.llm = ChatOllama(
                base_url=self.settings.llm.ollama_base_url,
                model=self.settings.llm.chat_model_name,
                temperature=self.settings.memory.summarization_temperature
            )
            logger.info("Initialized Ollama LLM for memory summarization")
        else:
            # Production: OpenAI
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(
                api_key=self.settings.llm.openai_api_key,
                model=self.settings.llm.chat_model_name,
                temperature=self.settings.memory.summarization_temperature
            )
            logger.info("Initialized OpenAI LLM for memory summarization")

    @contextmanager
    def _get_session(self) -> Session:
        """Context manager for database sessions.

        Automatically commits on success, rolls back on error.

        Yields:
            SQLAlchemy session
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database transaction failed: {e}", exc_info=True)
            raise
        finally:
            session.close()
```

### Get Session Memory (Load or Initialize)

```python
def get_session_memory(self, session_id: str) -> dict[str, Any]:
    """Load session memory or initialize if new.

    Returns memory structure even if session doesn't exist (default empty).
    This ensures calling code never receives None.

    Args:
        session_id: Unique session identifier

    Returns:
        Dict with 'summary' and 'messages' keys

    Example:
        >>> memory = service.get_session_memory("session-123")
        >>> print(memory)
        {"summary": "User asked about vacation...", "messages": ["User: ..."]}
    """
    if not session_id or not session_id.strip():
        raise ValueError("session_id cannot be empty")

    with self._get_session() as session:
        chat_session = session.query(ChatSession).filter_by(
            session_id=session_id
        ).first()

        if not chat_session:
            # New session - return default structure
            logger.info(f"Session {session_id} not found, returning empty memory")
            return {"summary": "", "messages": []}

        logger.info(
            f"Loaded memory for session {session_id}: "
            f"summary_length={len(chat_session.messages.get('summary', ''))}, "
            f"messages_count={len(chat_session.messages.get('messages', []))}"
        )
        return chat_session.messages
```

### Append Single Message

```python
def append_message(self, session_id: str, role: str, content: str) -> None:
    """Append a single message to session memory.

    Creates session if it doesn't exist. Automatically triggers summarization
    if message threshold is exceeded.

    Args:
        session_id: Unique session identifier
        role: Message role ("User" or "Assistant")
        content: Message content

    Raises:
        ValueError: If session_id is empty or role is invalid
    """
    if not session_id or not session_id.strip():
        raise ValueError("session_id cannot be empty")

    if role not in ("User", "Assistant"):
        raise ValueError(f"Invalid role: {role}. Must be 'User' or 'Assistant'")

    logger.info(f"Appending message to session {session_id}: {role}")

    with self._get_session() as session:
        # Load or create session
        chat_session = session.query(ChatSession).filter_by(
            session_id=session_id
        ).first()

        if not chat_session:
            # Create new session
            chat_session = ChatSession(
                session_id=session_id,
                messages={"summary": "", "messages": []}
            )
            session.add(chat_session)
            logger.info(f"Created new session: {session_id}")

        # Append message
        formatted_message = f"{role}: {content}"
        chat_session.messages["messages"].append(formatted_message)

        # Check if summarization needed
        message_count = len(chat_session.messages["messages"])
        threshold = self.settings.memory.message_threshold

        if message_count >= threshold:
            logger.info(
                f"Message threshold ({threshold}) reached for session {session_id}. "
                f"Triggering summarization."
            )
            self._summarize_session(chat_session, session)

        # Mark as modified (SQLAlchemy tracks JSONB changes)
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(chat_session, "messages")
```

### Append Multiple Messages (Batch)

```python
def append_messages(
    self,
    session_id: str,
    messages: list[tuple[str, str]]
) -> None:
    """Append multiple messages to session memory in a single transaction.

    More efficient than multiple append_message calls. Creates session if
    it doesn't exist. Automatically triggers summarization if threshold
    is exceeded.

    Args:
        session_id: Unique session identifier
        messages: List of (role, content) tuples

    Example:
        >>> service.append_messages("session-123", [
        ...     ("User", "What is the vacation policy?"),
        ...     ("Assistant", "You get 15 days annually."),
        ... ])

    Raises:
        ValueError: If session_id is empty or any role is invalid
    """
    if not session_id or not session_id.strip():
        raise ValueError("session_id cannot be empty")

    # Validate all roles before processing
    for role, _ in messages:
        if role not in ("User", "Assistant"):
            raise ValueError(f"Invalid role: {role}. Must be 'User' or 'Assistant'")

    logger.info(f"Appending {len(messages)} messages to session {session_id}")

    with self._get_session() as session:
        # Load or create session
        chat_session = session.query(ChatSession).filter_by(
            session_id=session_id
        ).first()

        if not chat_session:
            chat_session = ChatSession(
                session_id=session_id,
                messages={"summary": "", "messages": []}
            )
            session.add(chat_session)
            logger.info(f"Created new session: {session_id}")

        # Append all messages
        for role, content in messages:
            formatted_message = f"{role}: {content}"
            chat_session.messages["messages"].append(formatted_message)

        # Check if summarization needed
        message_count = len(chat_session.messages["messages"])
        threshold = self.settings.memory.message_threshold

        if message_count >= threshold:
            logger.info(
                f"Message threshold ({threshold}) reached for session {session_id}. "
                f"Triggering summarization."
            )
            self._summarize_session(chat_session, session)

        # Mark as modified
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(chat_session, "messages")
```

## Summarization Pattern

### When to Summarize

Summarization is triggered when:
1. Message count >= threshold (default: 6)
2. During an append operation (checked after each append)
3. Only if memory feature is enabled

**Never summarize**:
- During read operations (get_session_memory)
- If threshold not exceeded
- If feature disabled

### LLM Prompt Structure

```python
SUMMARIZATION_PROMPT_TEMPLATE = """You are summarizing a conversation between a user and an HR chatbot.

{existing_summary_section}

Recent Messages:
{messages}

Create a concise summary (maximum {max_length} tokens) that captures:
1. Key questions asked by the user
2. Important information provided by the assistant
3. Any unresolved questions or topics
4. User preferences or context mentioned

The summary will be used to maintain context for future messages in this conversation.

Summary:"""


def _build_summarization_prompt(
    self,
    existing_summary: str,
    messages: list[str]
) -> str:
    """Build prompt for LLM summarization.

    Args:
        existing_summary: Current summary (may be empty)
        messages: List of message strings to summarize

    Returns:
        Formatted prompt string
    """
    # Include existing summary section if present
    existing_summary_section = ""
    if existing_summary and existing_summary.strip():
        existing_summary_section = f"""Current Summary (from earlier in conversation):
{existing_summary}

"""

    # Format messages
    messages_text = "\n".join(messages)

    # Build full prompt
    prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(
        existing_summary_section=existing_summary_section,
        messages=messages_text,
        max_length=self.settings.memory.max_summary_length
    )

    return prompt
```

### Summarization Implementation

```python
def _summarize_session(self, chat_session: ChatSession, session: Session) -> None:
    """Generate LLM summary and update session.

    Internal method called when message threshold is exceeded.
    Merges existing summary with all current messages.

    Args:
        chat_session: ChatSession ORM object to update
        session: Active database session (for transaction)

    Note:
        This method modifies chat_session in place and clears messages.
        On error, falls back to keeping existing summary (graceful degradation).
    """
    try:
        existing_summary = chat_session.messages.get("summary", "")
        messages = chat_session.messages.get("messages", [])

        logger.info(
            f"Summarizing session {chat_session.session_id}: "
            f"existing_summary_length={len(existing_summary)}, "
            f"messages_to_summarize={len(messages)}"
        )

        # Build prompt
        prompt = self._build_summarization_prompt(existing_summary, messages)

        # Call LLM
        from langchain_core.messages import HumanMessage

        response = self.llm.invoke([HumanMessage(content=prompt)])
        new_summary = response.content.strip()

        # Validate summary
        if not new_summary or len(new_summary) < 10:
            logger.warning(
                f"Generated summary too short ({len(new_summary)} chars). "
                f"Keeping existing summary."
            )
            return

        # Update session
        chat_session.messages["summary"] = new_summary
        chat_session.messages["messages"] = []  # Clear messages after summarization

        logger.info(
            f"Summarization complete for session {chat_session.session_id}: "
            f"new_summary_length={len(new_summary)}, messages_cleared={len(messages)}"
        )

    except Exception as e:
        # Graceful degradation - keep existing summary, don't clear messages
        logger.error(
            f"Summarization failed for session {chat_session.session_id}: {e}",
            exc_info=True
        )
        logger.warning(
            "Continuing without summarization. "
            "Messages will accumulate until next successful summarization."
        )
```

## Memory Lifecycle Pattern

### Complete Flow: Append → Check → Summarize

```
User sends message
       ↓
API calls append_message(session_id, "User", message)
       ↓
MemoryService loads session from DB (or creates if new)
       ↓
Appends formatted message: "User: message content"
       ↓
Checks: len(messages) >= threshold?
       ↓
   Yes → Summarize
       ↓
       Build prompt with existing summary + all messages
       ↓
       Call LLM to generate new summary
       ↓
       Update: summary = new_summary, messages = []
       ↓
   No → Skip summarization
       ↓
Commit transaction
       ↓
Return to caller
```

### Example Usage in API

```python
# In chat endpoint, after graph execution:

from services.memory_service import MemoryService

async def update_conversation_memory(
    session_id: str,
    user_message: str,
    assistant_response: str
) -> None:
    """Update conversation memory with latest turn.

    Args:
        session_id: Unique session identifier
        user_message: User's question from this turn
        assistant_response: Assistant's answer from this turn
    """
    settings = get_settings()

    if not settings.memory.enabled:
        logger.debug("Memory disabled, skipping update")
        return

    try:
        memory_service = MemoryService()

        # Batch append (more efficient)
        memory_service.append_messages(
            session_id=session_id,
            messages=[
                ("User", user_message),
                ("Assistant", assistant_response)
            ]
        )

        logger.info(f"Memory updated for session: {session_id}")

    except Exception as e:
        # Don't fail the request if memory update fails
        logger.error(f"Memory update failed: {e}", exc_info=True)
        logger.warning("Continuing without memory update")


# Usage
await update_conversation_memory(
    session_id=request.session_id,
    user_message=request.message,
    assistant_response=final_state["final_answer"]
)
```

## Best Practices

### Do This

```python
# ✅ Use batch operations when appending multiple messages
memory_service.append_messages(session_id, [
    ("User", "Question 1"),
    ("Assistant", "Answer 1"),
    ("User", "Question 2"),
    ("Assistant", "Answer 2")
])

# ✅ Check if memory is enabled before processing
if settings.memory.enabled:
    memory_service.append_message(session_id, "User", message)

# ✅ Handle missing sessions gracefully
memory = memory_service.get_session_memory(session_id)
# Always returns dict, never None

# ✅ Use context managers for database sessions
with self._get_session() as session:
    # Operations here
    pass  # Auto-commit on success

# ✅ Log all memory operations with session ID
logger.info(f"Appending message to session {session_id}")

# ✅ Flag JSONB columns as modified after updates
from sqlalchemy.orm.attributes import flag_modified
flag_modified(chat_session, "messages")

# ✅ Use environment-aware LLM initialization
if settings.llm.is_local:
    llm = ChatOllama(...)
else:
    llm = ChatOpenAI(...)

# ✅ Validate inputs at function entry
if not session_id or not session_id.strip():
    raise ValueError("session_id cannot be empty")
```

### Don't Do This

```python
# ❌ Multiple individual appends (inefficient)
memory_service.append_message(session_id, "User", "Question 1")
memory_service.append_message(session_id, "Assistant", "Answer 1")
memory_service.append_message(session_id, "User", "Question 2")
# Use append_messages instead

# ❌ Return None for missing sessions
def get_session_memory(session_id: str) -> dict[str, Any] | None:
    session = query(ChatSession).first()
    return session.messages if session else None  # Bad!

# ❌ Catch exceptions and silence them
try:
    memory_service.append_message(...)
except:
    pass  # Error information lost

# ❌ Hardcode model names
llm = ChatOllama(model="mistral")  # Won't work in production

# ❌ Forget to mark JSONB as modified
chat_session.messages["messages"].append("new")
session.commit()  # Change may not persist

# ❌ Mutate default arguments
def get_memory(messages: list[str] = []) -> dict:  # Bug!
    messages.append("new")
    return {"messages": messages}

# ❌ Summarize on every message (expensive)
def append_message(...):
    # Append logic
    self._summarize_session(...)  # Don't always summarize!

# ❌ Store unformatted messages
chat_session.messages["messages"].append(content)  # Missing role prefix

# ❌ Use blocking operations in async context
async def update_memory():
    memory_service.append_message(...)  # This is blocking! Use run_in_executor
```

## Error Handling

### Session Not Found

```python
# ✅ Initialize with defaults (idempotent)
def get_session_memory(self, session_id: str) -> dict[str, Any]:
    with self._get_session() as session:
        chat_session = session.query(ChatSession).filter_by(
            session_id=session_id
        ).first()

        if not chat_session:
            return {"summary": "", "messages": []}  # Safe default

        return chat_session.messages
```

### Summarization Failure

```python
# ✅ Graceful degradation (keep messages)
def _summarize_session(self, chat_session: ChatSession, session: Session) -> None:
    try:
        # Summarization logic
        new_summary = self.llm.invoke(prompt).content
        chat_session.messages["summary"] = new_summary
        chat_session.messages["messages"] = []
    except Exception as e:
        logger.error(f"Summarization failed: {e}", exc_info=True)
        logger.warning("Keeping existing messages. Will retry on next append.")
        # Don't clear messages - let them accumulate for next attempt
```

### Database Transaction Failure

```python
# ✅ Automatic rollback with context manager
@contextmanager
def _get_session(self) -> Session:
    session = self.SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()  # Automatic
        logger.error(f"Transaction failed: {e}", exc_info=True)
        raise  # Re-raise after rollback
    finally:
        session.close()
```

### Invalid Input

```python
# ✅ Validate early, fail fast
def append_message(self, session_id: str, role: str, content: str) -> None:
    if not session_id or not session_id.strip():
        raise ValueError("session_id cannot be empty")

    if role not in ("User", "Assistant"):
        raise ValueError(f"Invalid role: {role}")

    if not content or not content.strip():
        raise ValueError("Message content cannot be empty")

    # Proceed with processing
```

### LLM API Timeout

```python
# ✅ Set timeout and handle gracefully
try:
    from langchain_core.messages import HumanMessage

    response = self.llm.invoke(
        [HumanMessage(content=prompt)],
        config={"timeout": 30}  # 30 second timeout
    )
    new_summary = response.content.strip()

except TimeoutError:
    logger.error("LLM summarization timed out")
    logger.warning("Keeping existing summary")
    return  # Don't clear messages

except Exception as e:
    logger.error(f"LLM call failed: {e}", exc_info=True)
    return  # Graceful degradation
```

## Configuration Reference

### Memory Settings

```python
from pydantic import BaseModel, Field


class MemoryConfig(BaseModel):
    """Configuration for conversation memory feature."""

    enabled: bool = Field(
        default=True,
        description="Enable conversation memory feature"
    )

    message_threshold: int = Field(
        default=6,
        ge=2,
        le=20,
        description="Number of messages before triggering summarization"
    )

    max_summary_length: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Maximum token length for conversation summaries"
    )

    summarization_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="LLM temperature for generating summaries (lower = more focused)"
    )
```

### Environment Variables

```bash
# Enable/disable memory
MEMORY_ENABLED=true

# Summarization threshold (messages)
MEMORY_MESSAGE_THRESHOLD=6

# Max summary length (tokens)
MEMORY_MAX_SUMMARY_LENGTH=500

# LLM temperature for summarization
MEMORY_SUMMARIZATION_TEMPERATURE=0.3
```

## Testing Patterns

### Unit Test: Memory Lifecycle

```python
def test_memory_lifecycle():
    """Test complete memory lifecycle with summarization."""
    service = MemoryService()
    session_id = f"test-{uuid4()}"

    # 1. New session returns empty
    memory = service.get_session_memory(session_id)
    assert memory == {"summary": "", "messages": []}

    # 2. Add messages below threshold
    service.append_messages(session_id, [
        ("User", "Q1"),
        ("Assistant", "A1"),
        ("User", "Q2"),
        ("Assistant", "A2")
    ])

    memory = service.get_session_memory(session_id)
    assert len(memory["messages"]) == 4
    assert memory["summary"] == ""

    # 3. Add more to exceed threshold (trigger summarization)
    service.append_messages(session_id, [
        ("User", "Q3"),
        ("Assistant", "A3")
    ])

    memory = service.get_session_memory(session_id)
    assert memory["summary"] != ""  # Summary generated
    assert len(memory["messages"]) == 0  # Messages cleared
```

### Integration Test: Error Handling

```python
def test_summarization_failure_graceful_degradation(monkeypatch):
    """Test memory continues working when summarization fails."""
    service = MemoryService()
    session_id = f"test-{uuid4()}"

    # Mock LLM to raise exception
    def mock_invoke(*args, **kwargs):
        raise Exception("LLM API error")

    monkeypatch.setattr(service.llm, "invoke", mock_invoke)

    # Add messages beyond threshold
    for i in range(4):
        service.append_message(session_id, "User", f"Q{i}")
        service.append_message(session_id, "Assistant", f"A{i}")

    # Should not crash, messages should still be stored
    memory = service.get_session_memory(session_id)
    assert len(memory["messages"]) == 8  # Not summarized due to error
    assert memory["summary"] == ""  # No summary
```

## Summary

**Key Takeaways**:
1. Use JSONB for flexible, simple storage
2. Summarize only when threshold exceeded (not every turn)
3. Always return defaults for missing sessions (never None)
4. Use environment-aware LLM selection
5. Handle failures gracefully (memory is non-critical)
6. Batch operations are more efficient than individual calls
7. Always use transactions for JSONB updates
8. Log all operations with session ID for traceability

**When to Use This Pattern**:
- Multi-turn chatbot conversations
- Session-based context retention
- Token-limited LLM contexts
- Proof-of-concept memory implementations

**When NOT to Use This Pattern**:
- Long-term user profiles (use separate user table)
- Full conversation history export (consider separate archive table)
- Real-time collaborative editing (use different architecture)
- Cross-session search (requires different indexing strategy)
