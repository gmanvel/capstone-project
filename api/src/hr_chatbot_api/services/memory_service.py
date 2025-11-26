"""Memory service for conversation management.

This module provides the MemoryService class that handles all conversation
memory operations including session CRUD, message persistence, and LLM-based
summarization of conversation history.
"""

import logging
from contextlib import contextmanager
from typing import Any

from langchain_core.messages import HumanMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.orm.attributes import flag_modified

from hr_chatbot_config import get_settings
from rag_pipeline.models import ChatSession

logger = logging.getLogger(__name__)

# Prompt template for LLM-based conversation summarization
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


class MemoryService:
    """Service for managing conversation memory.

    Handles session storage, message persistence, and LLM-based summarization
    of conversation history. Uses threshold-based summarization to keep memory
    size manageable while preserving context.

    The service stores conversation data in a PostgreSQL database using JSONB
    format for flexibility. When the number of messages exceeds a threshold,
    older messages are automatically summarized using an LLM to stay within
    token limits.

    Example:
        >>> service = MemoryService()
        >>> service.append_message("session-123", "User", "What is the policy?")
        >>> memory = service.get_session_memory("session-123")
        >>> print(memory["messages"])
        ["User: What is the policy?"]
    """

    def __init__(self):
        """Initialize memory service with settings and LLM.

        Sets up database connection, session factory, and environment-aware
        LLM for summarization (Ollama for development, OpenAI for production).
        """
        self.settings = get_settings()
        self.engine = create_engine(self.settings.database.connection_string)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self._init_llm()

    def _init_llm(self) -> None:
        """Initialize environment-aware LLM for summarization.

        Uses settings.llm.is_local to determine whether to use Ollama
        (development) or OpenAI (production). Temperature is set based
        on memory configuration for consistent, focused summaries.
        """
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
    def _get_session(self):
        """Context manager for database sessions.

        Automatically commits on success, rolls back on error, and always
        closes the session. This ensures transaction safety and prevents
        connection leaks.

        Yields:
            SQLAlchemy session

        Example:
            >>> with self._get_session() as session:
            ...     page = session.query(ChatSession).first()
            ...     # Auto-commit on success
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

    def get_session_memory(self, session_id: str) -> dict[str, Any]:
        """Load session memory or initialize if new.

        Returns memory structure even if session doesn't exist (default empty).
        This ensures calling code never receives None and can safely proceed.

        Args:
            session_id: Unique session identifier

        Returns:
            Dict with 'summary' (str) and 'messages' (list[str]) keys.
            Example: {"summary": "User asked about vacation...",
                     "messages": ["User: Can I carry over days?"]}

        Raises:
            ValueError: If session_id is empty or None
            SQLAlchemyError: If database query fails
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

    def append_message(self, session_id: str, role: str, content: str) -> None:
        """Append a single message to session memory.

        Creates session if it doesn't exist. Automatically triggers summarization
        if message threshold is exceeded after appending.

        Args:
            session_id: Unique session identifier
            role: Message role ("User" or "Assistant")
            content: Message content

        Raises:
            ValueError: If session_id is empty or role is invalid
            SQLAlchemyError: If database operation fails
        """
        if not session_id or not session_id.strip():
            raise ValueError("session_id cannot be empty")

        if role not in ("User", "Assistant"):
            raise ValueError(
                f"Invalid role: {role}. Must be 'User' or 'Assistant'"
            )

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
                    f"Message threshold ({threshold}) reached for session "
                    f"{session_id}. Triggering summarization."
                )
                self._summarize_session(chat_session, session)

            # Mark as modified (SQLAlchemy tracks JSONB changes)
            flag_modified(chat_session, "messages")

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
            SQLAlchemyError: If database operation fails
        """
        if not session_id or not session_id.strip():
            raise ValueError("session_id cannot be empty")

        # Validate all roles before processing
        for role, _ in messages:
            if role not in ("User", "Assistant"):
                raise ValueError(
                    f"Invalid role: {role}. Must be 'User' or 'Assistant'"
                )

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
                    f"Message threshold ({threshold}) reached for session "
                    f"{session_id}. Triggering summarization."
                )
                self._summarize_session(chat_session, session)

            # Mark as modified
            flag_modified(chat_session, "messages")

    def _build_summarization_prompt(
        self,
        existing_summary: str,
        messages: list[str]
    ) -> str:
        """Build prompt for LLM summarization.

        Combines existing summary (if any) with recent messages to create
        a prompt that instructs the LLM to generate an updated summary.

        Args:
            existing_summary: Current summary (may be empty)
            messages: List of message strings to summarize

        Returns:
            Formatted prompt string ready for LLM
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

    def _summarize_session(
        self,
        chat_session: ChatSession,
        session: Session
    ) -> None:
        """Generate LLM summary and update session.

        Internal method called when message threshold is exceeded.
        Merges existing summary with all current messages into a new
        summary, then clears the message buffer.

        Implements graceful degradation: if summarization fails, the
        existing summary is kept and messages are not cleared, allowing
        retry on the next append.

        Args:
            chat_session: ChatSession ORM object to update
            session: Active database session (for transaction)

        Note:
            This method modifies chat_session in place. On error, it logs
            the failure and returns without clearing messages.
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
                f"new_summary_length={len(new_summary)}, "
                f"messages_cleared={len(messages)}"
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

    def summarize_session(self, session_id: str) -> str:
        """Generate LLM summary of current conversation.

        Public method to manually trigger summarization. Useful for testing
        or forcing summarization before the threshold is reached.

        Args:
            session_id: Unique session identifier

        Returns:
            New summary text. If summarization fails, returns existing summary.

        Raises:
            ValueError: If session_id is empty
            SQLAlchemyError: If database operation fails
        """
        if not session_id or not session_id.strip():
            raise ValueError("session_id cannot be empty")

        with self._get_session() as session:
            chat_session = session.query(ChatSession).filter_by(
                session_id=session_id
            ).first()

            if not chat_session:
                logger.warning(f"Session {session_id} not found for summarization")
                return ""

            existing_summary = chat_session.messages.get("summary", "")

            # Trigger summarization
            self._summarize_session(chat_session, session)

            # Return updated summary (may be same as existing if failed)
            return chat_session.messages.get("summary", existing_summary)

    def clear_session(self, session_id: str) -> None:
        """Delete a session.

        Removes session and all its conversation history from the database.
        Idempotent - no error if session doesn't exist.

        Args:
            session_id: Unique session identifier

        Raises:
            ValueError: If session_id is empty
            SQLAlchemyError: If database operation fails
        """
        if not session_id or not session_id.strip():
            raise ValueError("session_id cannot be empty")

        with self._get_session() as session:
            chat_session = session.query(ChatSession).filter_by(
                session_id=session_id
            ).first()

            if not chat_session:
                logger.info(f"Session {session_id} not found, nothing to clear")
                return

            session.delete(chat_session)
            logger.info(f"Cleared session: {session_id}")
