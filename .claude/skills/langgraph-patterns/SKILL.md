# LangGraph Patterns

## Overview

This skill provides patterns for implementing multi-agent workflows using LangGraph.

**Scope**: Agent orchestration, state management, graph construction, and node implementation for the HR Chatbot's draft → critique workflow.

## Technology Recommendations

### Framework: LangGraph
- **Why**: Purpose-built for multi-agent workflows with explicit state management
- **Version**: 0.0.20+
- **Use for**: Agent orchestration, conditional routing, state transitions

### State Management: TypedDict
- **Why**: Type-safe, explicit schema, LangGraph native support
- **Critical**: State is immutable - nodes return new dicts, never mutate

### LLM Integration: LangChain
- **Why**: Seamless integration with LangGraph, environment-aware model switching
- **Use**: ChatOllama (dev), ChatOpenAI (prod)

## Core Principle: Immutable State

**CRITICAL: LangGraph state MUST be treated as immutable.**

```python
# ✅ CORRECT: Return new dict (partial update)
def draft_node(state: AgentState) -> dict[str, Any]:
    """Create draft answer."""
    draft = llm.invoke(...)
    
    # Return only changed fields - LangGraph merges automatically
    return {
        "draft_answer": draft.content,
        "confidence": 0.85
    }

# ✅ ALSO CORRECT: Return complete new state
def draft_node(state: AgentState) -> AgentState:
    """Create draft answer."""
    draft = llm.invoke(...)
    
    # Return entirely new dict
    return {
        "query": state["query"],
        "retrieved_docs": state["retrieved_docs"],
        "draft_answer": draft.content,
        "critique": None,
        "final_answer": "",
        "confidence": 0.85,
        "sources": []
    }

# ❌ WRONG: Mutate state
def draft_node(state: AgentState) -> AgentState:
    draft = llm.invoke(...)
    state["draft_answer"] = draft.content  # DON'T MUTATE!
    state["confidence"] = 0.85             # DON'T MUTATE!
    return state
```

**Why immutability matters:**
- LangGraph tracks state changes
- Enables debugging and replay
- Prevents side effects
- Makes state transitions explicit

## State Definition Pattern

### AgentState TypedDict

```python
from typing import TypedDict, Any

class AgentState(TypedDict):
    """State shared across all agent nodes.
    
    All fields must be explicitly typed and documented.
    LangGraph automatically merges partial updates.
    """
    # Input
    query: str                          # User's original question
    
    # Retrieved context
    retrieved_docs: list[dict[str, Any]]  # Documents from RAG pipeline
    
    # Agent outputs
    draft_answer: str | None            # Initial answer from Drafter
    critique: str | None                # Feedback from Critic
    final_answer: str                   # Final refined answer
    confidence: float                   # Confidence score (0.0 to 1.0)
    
    # Output metadata
    sources: list[dict[str, str]]       # Source metadata (title + url)
    escalated: bool                     # Whether escalated to HR
```

**Key Points:**
- Use `TypedDict` (not Pydantic BaseModel)
- Document each field with inline comments
- Use `| None` for optional fields (Python 3.10+)
- Include all fields that any node might read or write
- Keep state flat (avoid deep nesting)

### State Initialization

```python
def create_initial_state(query: str) -> AgentState:
    """Create initial state for agent workflow.
    
    Args:
        query: User question
        
    Returns:
        Initial state with query and empty fields
    """
    return {
        "query": query,
        "retrieved_docs": [],
        "draft_answer": None,
        "critique": None,
        "final_answer": "",
        "confidence": 0.0,
        "sources": [],
        "escalated": False
    }
```

## Agent Node Patterns

### Node Structure

```python
from langchain_core.messages import HumanMessage, SystemMessage
import logging

logger = logging.getLogger(__name__)

def node_name(state: AgentState) -> dict[str, Any]:
    """Brief description of what this node does.
    
    Args:
        state: Current agent state with required fields
        
    Returns:
        Dict with updated state fields (partial update)
        
    Raises:
        ValueError: If required state fields missing
        Exception: If LLM invocation fails
    """
    logger.info("Node execution started")
    
    try:
        # 1. Extract needed data from state
        input_data = state["some_field"]
        
        # 2. Perform node logic
        result = process(input_data)
        
        # 3. Return partial state update
        return {
            "output_field": result
        }
    
    except Exception as e:
        logger.error(f"Node failed: {e}", exc_info=True)
        raise
```

### Retrieval Node

```python
def retrieve_node(state: AgentState) -> dict[str, Any]:
    """Retrieve relevant documents from RAG pipeline.
    
    Queries vector store with user's question and returns
    top-k most relevant document chunks.
    
    Args:
        state: Current state with 'query' field
        
    Returns:
        Dict with 'retrieved_docs' list
    """
    logger.info(f"Retrieving documents for query: {state['query'][:50]}...")
    
    from rag_pipeline import RAGPipeline
    
    rag = RAGPipeline()
    
    try:
        docs = rag.retrieve(
            query=state["query"],
            top_k=5,
            score_threshold=0.7
        )
        
        logger.info(f"Retrieved {len(docs)} documents")
        
        return {"retrieved_docs": docs}
    
    except Exception as e:
        logger.error(f"Retrieval failed: {e}", exc_info=True)
        # Return empty docs on failure (graceful degradation)
        return {"retrieved_docs": []}
```

### Draft Node

```python
from hr_chatbot_config import get_settings

def draft_node(state: AgentState) -> dict[str, Any]:
    """Create initial draft answer from retrieved context.
    
    Uses LLM to generate answer based only on retrieved documents.
    Estimates confidence based on number and quality of sources.
    
    Args:
        state: Current state with 'query' and 'retrieved_docs'
        
    Returns:
        Dict with 'draft_answer' and 'confidence'
    """
    logger.info("Creating draft answer")
    
    # Get environment-aware LLM
    settings = get_settings()
    if settings.llm.is_local:
        from langchain_community.chat_models import ChatOllama
        llm = ChatOllama(
            base_url=settings.llm.ollama_base_url,
            model=settings.llm.chat_model_name,
            temperature=0.3
        )
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            api_key=settings.llm.openai_api_key,
            model=settings.llm.chat_model_name,
            temperature=0.3
        )
    
    # Build context from retrieved docs
    if not state["retrieved_docs"]:
        logger.warning("No documents retrieved, creating low-confidence response")
        return {
            "draft_answer": "I couldn't find relevant information to answer your question.",
            "confidence": 0.2
        }
    
    context = "\n\n".join([
        f"Source: {doc['title']}\n{doc['text']}"
        for doc in state["retrieved_docs"]
    ])
    
    # Create prompt
    messages = [
        SystemMessage(content="""You are an HR assistant. Answer questions based ONLY on the provided context.
If the context doesn't contain enough information, say so.
Be concise and accurate."""),
        HumanMessage(content=f"""Context from HR documents:

{context}

Question: {state['query']}

Provide a clear answer based on the context above.""")
    ]
    
    # Invoke LLM
    try:
        response = llm.invoke(messages)
        draft = response.content
        
        # Estimate confidence based on retrieval quality
        # Could be improved with LLM self-assessment
        avg_score = sum(doc["score"] for doc in state["retrieved_docs"]) / len(state["retrieved_docs"])
        confidence = min(avg_score, 0.95)  # Cap at 0.95
        
        logger.info(f"Draft created with confidence: {confidence:.2f}")
        
        return {
            "draft_answer": draft,
            "confidence": confidence
        }
    
    except Exception as e:
        logger.error(f"Draft generation failed: {e}", exc_info=True)
        return {
            "draft_answer": "I encountered an error generating the answer.",
            "confidence": 0.0
        }
```

### Critique Node

```python
def critique_node(state: AgentState) -> dict[str, Any]:
    """Critique and refine the draft answer.
    
    Reviews draft for accuracy, completeness, and clarity.
    Produces improved final answer.
    
    Args:
        state: Current state with 'query', 'retrieved_docs', 'draft_answer'
        
    Returns:
        Dict with 'critique', 'final_answer', 'confidence', 'sources'
    """
    logger.info("Critiquing and refining draft")
    
    # Get environment-aware LLM
    settings = get_settings()
    if settings.llm.is_local:
        from langchain_community.chat_models import ChatOllama
        llm = ChatOllama(
            base_url=settings.llm.ollama_base_url,
            model=settings.llm.chat_model_name,
            temperature=0.2
        )
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            api_key=settings.llm.openai_api_key,
            model=settings.llm.chat_model_name,
            temperature=0.2
        )
    
    # Build context
    context = "\n\n".join([
        f"{doc['text']}"
        for doc in state["retrieved_docs"]
    ])
    
    # Create critique prompt
    messages = [
        SystemMessage(content="""You are a senior HR advisor reviewing a junior assistant's answer.
Your job is to improve the answer by:
1. Ensuring accuracy against the source context
2. Adding missing important details
3. Improving clarity and professionalism
4. Keeping it concise

Provide only the improved answer, not a critique."""),
        HumanMessage(content=f"""Original Question: {state['query']}

Draft Answer: {state['draft_answer']}

Source Context:
{context}

Provide an improved, final answer.""")
    ]
    
    # Invoke LLM
    try:
        response = llm.invoke(messages)
        final_answer = response.content
        
        # Extract sources for citation
        sources = [
            {"title": doc["title"], "url": doc["url"]}
            for doc in state["retrieved_docs"]
        ]
        
        # Remove duplicates
        seen = set()
        unique_sources = []
        for source in sources:
            key = source["url"]
            if key not in seen:
                seen.add(key)
                unique_sources.append(source)
        
        logger.info(f"Refined answer with {len(unique_sources)} sources")
        
        return {
            "critique": "Applied improvements for accuracy and clarity",
            "final_answer": final_answer,
            "confidence": state["confidence"],  # Keep same confidence
            "sources": unique_sources
        }
    
    except Exception as e:
        logger.error(f"Critique failed: {e}", exc_info=True)
        # Fall back to draft
        return {
            "critique": "Critique failed, using draft",
            "final_answer": state["draft_answer"],
            "confidence": state["confidence"],
            "sources": []
        }
```

### Tool Node (Notify HR)

```python
def notify_hr_node(state: AgentState) -> dict[str, Any]:
    """Escalate low-confidence question to HR team.
    
    Logs question for HR review and optionally sends notification.
    Returns user-friendly escalation message.
    
    Args:
        state: Current state with 'query' and 'confidence'
        
    Returns:
        Dict with 'final_answer' and 'escalated' flag
    """
    logger.warning(f"Escalating low-confidence question: {state['query']}")
    
    # Log for HR review
    logger.info(
        "HR escalation",
        extra={
            "query": state["query"],
            "confidence": state["confidence"],
            "retrieved_docs_count": len(state["retrieved_docs"])
        }
    )
    
    # TODO: Send notification (email, Slack, etc.)
    # send_hr_notification(state["query"])
    
    return {
        "final_answer": (
            "I couldn't find a confident answer to your question. "
            "Your question has been forwarded to the HR team, and they will respond shortly."
        ),
        "escalated": True,
        "sources": []
    }
```

## Conditional Edge Patterns

### Simple Condition

```python
def should_notify_hr(state: AgentState) -> str:
    """Determine routing based on confidence score.
    
    Routes to HR notification if confidence below threshold,
    otherwise ends workflow.
    
    Args:
        state: Current state with 'confidence' field
        
    Returns:
        "notify" if low confidence, "end" if high confidence
    """
    CONFIDENCE_THRESHOLD = 0.7
    
    if state["confidence"] < CONFIDENCE_THRESHOLD:
        logger.warning(
            f"Low confidence ({state['confidence']:.2f}), routing to HR notification"
        )
        return "notify"
    
    logger.info(f"High confidence ({state['confidence']:.2f}), ending workflow")
    return "end"
```

### Complex Condition

```python
def route_based_on_docs(state: AgentState) -> str:
    """Route based on document retrieval results.
    
    Routes to:
    - "draft" if documents found
    - "no_context" if no documents
    - "low_quality" if docs exist but low scores
    
    Args:
        state: Current state with 'retrieved_docs'
        
    Returns:
        Route name
    """
    docs = state["retrieved_docs"]
    
    if not docs:
        logger.warning("No documents retrieved, routing to no_context handler")
        return "no_context"
    
    avg_score = sum(doc["score"] for doc in docs) / len(docs)
    
    if avg_score < 0.5:
        logger.warning(f"Low quality docs (avg score: {avg_score:.2f})")
        return "low_quality"
    
    logger.info(f"Good docs (avg score: {avg_score:.2f}), proceeding to draft")
    return "draft"
```

## Graph Construction Pattern

### Basic Linear Graph

```python
from langgraph.graph import StateGraph, END

def build_agent_graph() -> StateGraph:
    """Build the agent workflow graph.
    
    Flow: retrieve → draft → critique → confidence check → (end | notify_hr)
    
    Returns:
        Compiled LangGraph
    """
    # Create graph with state schema
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("draft", draft_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("notify_hr", notify_hr_node)
    
    # Set entry point
    workflow.set_entry_point("retrieve")
    
    # Add edges (deterministic flow)
    workflow.add_edge("retrieve", "draft")
    workflow.add_edge("draft", "critique")
    
    # Add conditional edge (branching)
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
```

### Graph with Error Handling

```python
def build_robust_agent_graph() -> StateGraph:
    """Build agent graph with error handling nodes.
    
    Includes error recovery nodes for graceful degradation.
    """
    workflow = StateGraph(AgentState)
    
    # Add normal nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("draft", draft_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("notify_hr", notify_hr_node)
    
    # Add error handling node
    workflow.add_node("error_handler", error_handler_node)
    
    # Set entry point
    workflow.set_entry_point("retrieve")
    
    # Add conditional edge from retrieve
    workflow.add_conditional_edges(
        "retrieve",
        route_based_on_docs,
        {
            "draft": "draft",
            "no_context": "notify_hr",
            "low_quality": "error_handler"
        }
    )
    
    workflow.add_edge("draft", "critique")
    workflow.add_conditional_edges(
        "critique",
        should_notify_hr,
        {
            "notify": "notify_hr",
            "end": END
        }
    )
    workflow.add_edge("notify_hr", END)
    workflow.add_edge("error_handler", "notify_hr")
    
    return workflow.compile()
```

## Graph Invocation Patterns

### Synchronous Invocation

```python
def process_query(query: str) -> dict[str, Any]:
    """Process user query through agent workflow.
    
    Args:
        query: User question
        
    Returns:
        Final state with answer and metadata
    """
    logger.info(f"Processing query: {query[:50]}...")
    
    # Build graph
    graph = build_agent_graph()
    
    # Create initial state
    initial_state = create_initial_state(query)
    
    # Invoke graph
    try:
        final_state = graph.invoke(initial_state)
        
        logger.info(
            f"Query processed: confidence={final_state['confidence']:.2f}, "
            f"escalated={final_state.get('escalated', False)}"
        )
        
        return final_state
    
    except Exception as e:
        logger.error(f"Graph invocation failed: {e}", exc_info=True)
        raise
```

### Async Invocation

```python
async def process_query_async(query: str) -> dict[str, Any]:
    """Process user query asynchronously.
    
    Args:
        query: User question
        
    Returns:
        Final state with answer and metadata
    """
    logger.info(f"Processing query async: {query[:50]}...")
    
    graph = build_agent_graph()
    initial_state = create_initial_state(query)
    
    try:
        final_state = await graph.ainvoke(initial_state)
        
        logger.info("Async query processed successfully")
        
        return final_state
    
    except Exception as e:
        logger.error(f"Async graph invocation failed: {e}", exc_info=True)
        raise
```

### Streaming Invocation

```python
def process_query_streaming(query: str):
    """Process query with streaming state updates.
    
    Yields state after each node execution.
    Useful for real-time UI updates.
    
    Args:
        query: User question
        
    Yields:
        State snapshots after each node
    """
    logger.info(f"Processing query with streaming: {query[:50]}...")
    
    graph = build_agent_graph()
    initial_state = create_initial_state(query)
    
    try:
        for state in graph.stream(initial_state):
            logger.debug(f"State update: {list(state.keys())}")
            yield state
    
    except Exception as e:
        logger.error(f"Streaming failed: {e}", exc_info=True)
        raise
```

## Testing Patterns

### Unit Test Node

```python
import pytest

def test_draft_node():
    """Test draft node with mock state."""
    state = {
        "query": "What is the vacation policy?",
        "retrieved_docs": [
            {
                "text": "Employees get 15 days vacation.",
                "title": "Vacation Policy",
                "score": 0.9,
                "url": "https://..."
            }
        ],
        "draft_answer": None,
        "critique": None,
        "final_answer": "",
        "confidence": 0.0,
        "sources": [],
        "escalated": False
    }
    
    result = draft_node(state)
    
    assert "draft_answer" in result
    assert "confidence" in result
    assert result["confidence"] > 0
    assert len(result["draft_answer"]) > 0
```

### Integration Test Graph

```python
def test_full_workflow():
    """Test complete agent workflow."""
    graph = build_agent_graph()
    
    initial_state = create_initial_state("What is the vacation policy?")
    
    final_state = graph.invoke(initial_state)
    
    assert "final_answer" in final_state
    assert "sources" in final_state
    assert final_state["confidence"] >= 0.0
    assert final_state["confidence"] <= 1.0
```

## Best Practices

### ✅ Do This

```python
# 1. Return partial state updates (only changed fields)
def node(state: AgentState) -> dict[str, Any]:
    return {"field": "new_value"}

# 2. Log node execution
logger.info("Node started")

# 3. Handle errors gracefully
try:
    result = risky_operation()
except Exception as e:
    logger.error(f"Failed: {e}", exc_info=True)
    return {"field": "fallback_value"}

# 4. Validate state inputs
if not state["required_field"]:
    raise ValueError("required_field is missing")

# 5. Use environment-aware LLM
if settings.llm.is_local:
    llm = ChatOllama(...)
else:
    llm = ChatOpenAI(...)

# 6. Document node behavior
def node(state: AgentState) -> dict[str, Any]:
    """Clear description of what node does."""
    pass
```

### ❌ Don't Do This

```python
# 1. Don't mutate state
def node(state: AgentState) -> AgentState:
    state["field"] = "value"  # DON'T MUTATE
    return state

# 2. Don't skip logging
def node(state):
    # No logs - hard to debug
    return {"field": "value"}

# 3. Don't hardcode LLM
def node(state):
    llm = ChatOllama(model="llama3.1")  # Won't work in prod
    
# 4. Don't catch exceptions silently
try:
    operation()
except:
    pass  # Error lost

# 5. Don't skip type hints
def node(state):  # Missing types
    return {"field": "value"}

# 6. Don't access state fields without checking
def node(state):
    value = state["maybe_missing"]  # Could KeyError
```

## Common Patterns Summary

| Pattern | Use Case | Implementation |
|---------|----------|----------------|
| Retrieval Node | Fetch context | Call RAG pipeline, return docs |
| LLM Node | Generate text | Invoke LLM with prompt, return response |
| Tool Node | External action | Call tool/API, return result |
| Conditional Edge | Branching logic | Return route string based on state |
| Error Handler | Graceful failure | Catch errors, return fallback |
| State Validation | Input checking | Validate required fields, raise if missing |

## Performance Tips

1. **Lazy LLM initialization**: Create LLM once, reuse across invocations
2. **Batch embeddings**: Generate embeddings in batches (done in RAG pipeline)
3. **Timeout handling**: Set LLM timeouts to prevent hanging
4. **Caching**: Cache frequent queries (future enhancement)
5. **Parallel nodes**: Use LangGraph parallel execution for independent nodes