"""Chat endpoint router for question answering.

This module provides the /chat endpoint that processes user questions
through a multi-agent RAG workflow with LangGraph.
"""

import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from hr_chatbot_config import get_settings

from ..agents.graph import build_agent_graph, create_initial_state
from ..models import ChatRequest, ChatResponse, SourceMetadata

logger = logging.getLogger(__name__)

router = APIRouter()


async def update_conversation_memory(
    session_id: str,
    user_message: str,
    assistant_response: str
) -> None:
    """Update conversation memory with the latest turn.

    Appends both the user message and assistant response to the session memory.
    Automatically triggers summarization if the message threshold is exceeded.

    This runs asynchronously after the graph completes to avoid blocking
    the response to the user.

    Args:
        session_id: Unique session identifier
        user_message: User's question/message from this turn
        assistant_response: Assistant's answer from this turn

    Raises:
        Exception: Logs but does not raise - memory update failures are non-critical
    """
    logger.info(f"Updating memory for session: {session_id}")

    settings = get_settings()

    # Check if memory is enabled
    if not settings.memory.enabled:
        logger.debug("Memory disabled, skipping update")
        return

    try:
        from ..services.memory_service import MemoryService

        memory_service = MemoryService()

        # Append both messages as a batch (more efficient)
        memory_service.append_messages(
            session_id=session_id,
            messages=[
                ("User", user_message),
                ("Assistant", assistant_response)
            ]
        )

        logger.info(f"Memory updated successfully for session: {session_id}")

    except Exception as e:
        # Don't fail the request if memory update fails
        logger.error(
            f"Failed to update memory for session {session_id}: {e}",
            exc_info=True
        )
        logger.warning("Continuing without memory update (non-critical failure)")


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process user question with multi-agent reasoning.

    Executes a LangGraph workflow with the following steps:
    1. Retrieve relevant documents from vector store
    2. Draft agent creates initial answer from context
    3. Critique agent refines and improves the answer
    4. If confidence < 0.7, escalate to HR team
    5. Update conversation memory (if enabled)

    Args:
        request: Chat request containing user message and optional session_id

    Returns:
        ChatResponse with AI-generated answer, sources, confidence, and metadata

    Raises:
        HTTPException 400: Invalid request (e.g., empty message)
        HTTPException 500: Internal processing error
    """
    logger.info(f"Chat request received: {request.message[:50]}...")

    # Generate or use provided session ID
    session_id = request.session_id or str(uuid4())

    try:
        # Build agent graph
        graph = build_agent_graph()

        # Create initial state (validates query)
        initial_state = create_initial_state(request.message, session_id)

        # Invoke agent workflow
        final_state = graph.invoke(initial_state)

        # Convert sources from dict to SourceMetadata objects
        sources = [
            SourceMetadata(**src)
            for src in final_state.get("sources", [])
        ]

        # Build response from final state
        response = ChatResponse(
            response=final_state["final_answer"],
            sources=sources,
            session_id=session_id,
            confidence=final_state.get("confidence", 0.0),
            escalated=final_state.get("escalated", False)
        )

        logger.info(
            f"Chat completed: confidence={response.confidence:.2f}, "
            f"escalated={response.escalated}"
        )

        # Update conversation memory (non-blocking, errors logged only)
        await update_conversation_memory(
            session_id=session_id,
            user_message=request.message,
            assistant_response=response.response
        )

        return response

    except ValueError as e:
        # Client error (invalid input)
        logger.warning(f"Invalid chat request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    except Exception as e:
        # Server error
        logger.error(f"Chat processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat request"
        )
