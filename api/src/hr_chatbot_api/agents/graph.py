"""LangGraph agent workflow construction.

This module defines the multi-agent workflow graph with LLM-powered node
implementations. The graph orchestrates: retrieve → draft → critique →
confidence check → (end | notify_hr).
"""

import logging
from typing import Any

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from hr_chatbot_config import get_settings
from rag_pipeline import RAGPipeline

from .agent_state import AgentState

logger = logging.getLogger(__name__)


# ========== HELPER FUNCTIONS ==========


def _get_llm():
    """Get environment-aware LLM instance.

    Returns Ollama (dev) or OpenAI (prod) based on settings.

    Returns:
        ChatOllama or ChatOpenAI instance
    """
    settings = get_settings()

    if settings.llm.is_local:
        logger.info("Using Ollama for LLM")
        return ChatOllama(
            base_url=settings.llm.ollama_base_url,
            model=settings.llm.chat_model_name,
            temperature=0.3
        )
    else:
        logger.info("Using OpenAI for LLM")
        return ChatOpenAI(
            api_key=settings.llm.openai_api_key,
            model=settings.llm.chat_model_name,
            temperature=0.3
        )


# ========== NODE FUNCTIONS ==========


def retrieve_memory_node(state: AgentState) -> dict[str, Any]:
    """Load conversation memory for the current session.

    Retrieves existing conversation history (summary + recent messages) from
    the database and injects it into the agent state. If the session is new
    or memory is disabled, returns empty memory fields.

    This node runs early in the workflow to ensure all subsequent nodes
    have access to conversational context.

    Args:
        state: Current agent state with 'session_id' field

    Returns:
        Dict with 'memory_summary' and 'memory_messages' fields

    Raises:
        ValueError: If session_id is missing or invalid
    """
    logger.info(f"Retrieving memory for session: {state['session_id']}")

    settings = get_settings()

    # Check if memory is enabled
    if not settings.memory.enabled:
        logger.info("Memory feature disabled, returning empty memory")
        return {
            "memory_summary": "",
            "memory_messages": []
        }

    # Validate session_id
    session_id = state.get("session_id")
    if not session_id or not session_id.strip():
        raise ValueError("session_id is required for memory retrieval")

    try:
        # Load memory from database
        from ..services.memory_service import MemoryService

        memory_service = MemoryService()
        memory = memory_service.get_session_memory(session_id)

        logger.info(
            f"Memory loaded: summary_length={len(memory['summary'])}, "
            f"messages_count={len(memory['messages'])}"
        )

        return {
            "memory_summary": memory["summary"],
            "memory_messages": memory["messages"]
        }

    except Exception as e:
        # Graceful degradation: if memory loading fails, proceed without it
        logger.error(f"Failed to load memory for session {session_id}: {e}", exc_info=True)
        logger.warning("Proceeding without conversation memory")
        return {
            "memory_summary": "",
            "memory_messages": []
        }


def retrieve_node(state: AgentState) -> dict[str, Any]:
    """Retrieve relevant documents from RAG pipeline.

    Queries vector store with user's question and returns top-k
    most relevant document chunks.

    Args:
        state: Current agent state with query

    Returns:
        Dict with retrieved_docs list
    """
    logger.info(f"Retrieving documents for query: {state['query'][:50]}...")

    rag = RAGPipeline()

    try:
        documents = rag.retrieve(
            query=state["query"],
            top_k=5,
            score_threshold=0.7
        )

        if not documents:
            logger.warning("No documents retrieved for query. Trying lower score threshold")
            documents = rag.retrieve(
                query=state["query"],
                top_k=5,
                score_threshold=0.6
            )

        # Convert Documents to dicts for state
        retrieved_docs = []
        for doc in documents:
            retrieved_docs.append({
                "text": doc.page_content,
                "title": doc.metadata["title"],
                "url": doc.metadata["url"],
                "score": doc.metadata["score"]
            })

        if not retrieved_docs:
            logger.warning("No documents retrieved for query")
        else:
            logger.info(f"Retrieved {len(retrieved_docs)} documents")

        return {"retrieved_docs": retrieved_docs}

    except Exception as e:
        logger.error(f"Retrieval failed: {e}", exc_info=True)
        # Graceful degradation: return empty docs
        return {"retrieved_docs": []}


def draft_node(state: AgentState) -> dict[str, Any]:
    """Create draft answer from retrieved context.

    Uses LLM to generate initial answer based on retrieved documents
    and conversation memory. Calculates confidence from retrieval scores.

    Args:
        state: Current state with query, retrieved_docs, and memory fields

    Returns:
        Dict with draft_answer and confidence
    """
    logger.info("Generating draft answer")

    llm = _get_llm()
    query = state["query"]
    retrieved_docs = state["retrieved_docs"]

    # Handle case with no retrieved documents
    if not retrieved_docs:
        logger.warning("No documents available for drafting answer")
        return {
            "draft_answer": "I couldn't find relevant information to answer your question.",
            "confidence": 0.2
        }

    # Build memory context
    memory_context = ""
    if state.get("memory_summary") or state.get("memory_messages"):
        memory_context = "\n\n=== Conversation History ===\n"

        if state["memory_summary"]:
            memory_context += f"Summary of earlier conversation:\n{state['memory_summary']}\n\n"

        if state["memory_messages"]:
            memory_context += "Recent messages:\n"
            memory_context += "\n".join(state["memory_messages"])
            memory_context += "\n"

    # Build context from retrieved documents
    context = ""
    for doc in retrieved_docs:
        context += f"Source: {doc['title']}\n{doc['text']}\n\n"

    # Create messages for LLM
    messages = [
        SystemMessage(content=(
            "You are an HR assistant. Answer questions based on:\n"
            "1. The conversation history (if provided)\n"
            "2. The retrieved HR documents (context)\n\n"
            "If the user asks a follow-up question, use the conversation history to understand the context."
        )),
        HumanMessage(content=(
            f"{memory_context}"
            f"Retrieved HR Documents:\n{context}\n"
            f"Current Question: {query}\n\n"
            f"Provide a clear, context-aware answer."
        ))
    ]

    # Invoke LLM
    response = llm.invoke(messages)
    draft = response.content

    # Calculate confidence from average retrieval score
    avg_score = sum(doc["score"] for doc in retrieved_docs) / len(retrieved_docs)
    confidence = min(avg_score, 0.95)  # Cap at 0.95

    logger.info(f"Draft created with confidence: {confidence:.2f}")

    return {
        "draft_answer": draft,
        "confidence": confidence
    }


def critique_node(state: AgentState) -> dict[str, Any]:
    """Critique and refine draft answer.

    Uses LLM as senior HR advisor to review and improve the draft answer,
    considering conversation history and ensuring accuracy, completeness,
    and professionalism.

    Args:
        state: Current state with draft_answer and memory fields

    Returns:
        Dict with critique, final_answer, confidence, sources
    """
    logger.info("Refining draft answer")

    llm = _get_llm()
    query = state["query"]
    retrieved_docs = state["retrieved_docs"]
    draft_answer = state["draft_answer"]

    # Build memory context
    memory_context = ""
    if state.get("memory_summary") or state.get("memory_messages"):
        memory_context = "\n\n=== Conversation History ===\n"

        if state["memory_summary"]:
            memory_context += f"Summary: {state['memory_summary']}\n\n"

        if state["memory_messages"]:
            memory_context += "Recent:\n" + "\n".join(state["memory_messages"]) + "\n"

    # Build context from retrieved documents
    context = ""
    for doc in retrieved_docs:
        context += f"Source: {doc['title']}\n{doc['text']}\n\n"

    # Create messages for LLM
    messages = [
        SystemMessage(content=(
            "You are a senior HR advisor reviewing an assistant's answer. "
            "Improve the answer considering:\n"
            "1. The conversation history (for context)\n"
            "2. The source documents (for accuracy)\n"
            "3. The original question\n\n"
            "Ensure the answer makes sense given the full conversation context. "
            "Provide only the improved answer, not a critique."
        )),
        HumanMessage(content=(
            f"{memory_context}"
            f"Original Question: {query}\n\n"
            f"Draft Answer: {draft_answer}\n\n"
            f"Source Context:\n{context}\n"
            f"Provide an improved, context-aware answer."
        ))
    ]

    # Invoke LLM
    response = llm.invoke(messages)
    final_answer = response.content

    # Build sources list and remove duplicates by URL
    seen_urls = set()
    sources = []
    for doc in retrieved_docs:
        if doc["url"] not in seen_urls:
            sources.append({
                "title": doc["title"],
                "url": doc["url"]
            })
            seen_urls.add(doc["url"])

    # Keep confidence from draft (don't recalculate)
    confidence = state["confidence"]

    logger.info(f"Refined answer with {len(sources)} sources")

    return {
        "critique": "Applied improvements",
        "final_answer": final_answer,
        "confidence": confidence,
        "sources": sources
    }


def notify_hr_node(state: AgentState) -> dict[str, Any]:
    """Escalate question to HR team.

    Logs the low-confidence question for HR review and returns
    a message indicating escalation to the user.

    Args:
        state: Current state with query

    Returns:
        Dict with final_answer and escalated flag
    """
    query = state["query"]
    confidence = state["confidence"]

    logger.warning(f"Escalating low-confidence question: {query}")
    logger.info(
        "Low confidence question escalated",
        extra={"query": query, "confidence": confidence}
    )

    # TODO: Send notification (email, Slack, etc.)

    return {
        "final_answer": (
            "I couldn't find a confident answer to your question. "
            "Your question has been forwarded to the HR team, "
            "and they will respond shortly."
        ),
        "escalated": True,
        "sources": []
    }


# ========== CONDITIONAL EDGE ==========

# Confidence threshold for HR escalation
CONFIDENCE_THRESHOLD = 0.6


def should_notify_hr(state: AgentState) -> str:
    """Determine if question should be escalated to HR.

    Checks if confidence score is below threshold and routes
    to HR notification if needed.

    Args:
        state: Current state with confidence

    Returns:
        "notify" if low confidence, "end" if high confidence
    """
    confidence = state["confidence"]

    if confidence < CONFIDENCE_THRESHOLD:
        logger.warning(
            f"Low confidence ({confidence:.2f}), routing to HR notification"
        )
        return "notify"
    else:
        logger.info(
            f"High confidence ({confidence:.2f}), ending workflow"
        )
        return "end"


# ========== GRAPH CONSTRUCTION ==========


def build_agent_graph() -> StateGraph:
    """Build and compile the agent workflow graph.

    Creates a LangGraph workflow with the following flow:
    retrieve_memory → retrieve → draft → critique → confidence_check → (end | notify_hr)

    The graph uses AgentState for type safety and automatic state merging.

    Returns:
        Compiled LangGraph workflow ready for execution
    """
    logger.info("Building agent graph")

    # Create graph with state schema
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("retrieve_memory", retrieve_memory_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("draft", draft_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("notify_hr", notify_hr_node)

    # Set entry point
    workflow.set_entry_point("retrieve_memory")

    # Add edges
    workflow.add_edge("retrieve_memory", "retrieve")
    workflow.add_edge("retrieve", "draft")
    workflow.add_edge("draft", "critique")

    # Add conditional edge based on confidence
    workflow.add_conditional_edges(
        "critique",
        should_notify_hr,
        {
            "notify": "notify_hr",
            "end": END
        }
    )

    # notify_hr leads to END
    workflow.add_edge("notify_hr", END)

    # Compile graph
    graph = workflow.compile()

    logger.info("Agent graph compiled successfully")

    return graph


def create_initial_state(query: str, session_id: str) -> AgentState:
    """Create initial state for agent workflow.

    Validates the query and initializes all state fields with empty or
    default values.

    Args:
        query: User question to process
        session_id: Session identifier for conversation tracking (default: empty)

    Returns:
        Initial AgentState dict ready for graph execution

    Raises:
        ValueError: If query is empty or whitespace-only
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    logger.info(f"Initial state created for query: {query[:50]}...")

    return {
        "query": query.strip(),
        "session_id": session_id,
        "memory_summary": "",
        "memory_messages": [],
        "retrieved_docs": [],
        "draft_answer": None,
        "critique": None,
        "final_answer": "",
        "confidence": 0.0,
        "sources": [],
        "escalated": False
    }
