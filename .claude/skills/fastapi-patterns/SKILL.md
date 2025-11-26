# FastAPI Patterns

## Overview

This skill provides patterns for implementing FastAPI endpoints for the HR Chatbot API service.

**Scope**: API structure, endpoint patterns, request/response models, error handling, dependency injection, and middleware.

## Technology Recommendations

### Framework: FastAPI
- **Version**: 0.109.0+
- **Why**: Async support, automatic OpenAPI docs, Pydantic validation, type safety

### Server: Uvicorn
- **Why**: ASGI server, async support, hot reload in development
- **Usage**: `uvicorn api.main:app --reload --port 8000`

### Validation: Pydantic
- **Why**: Built into FastAPI, type-safe, automatic validation
- **Version**: 2.0+

### CORS: FastAPI middleware
- **When**: If frontend hosted on different domain
- **Default**: Not needed for this project (API-only)

## Project Structure

```
api/
├── src/
│   ├
│   ├── main.py              # FastAPI app & startup
│   ├── routers/             # Endpoint routers
│       │   ├── __init__.py
│       │   ├── chat.py          # /chat endpoint
│       │   └── setup.py         # /setup endpoint
│   ├── agents/              # LangGraph agents
│       │   ├── __init__.py
│       │   └── graph.py         # Agent graph
│       ├── models.py            # Request/response models
│       └── dependencies.py      # Shared dependencies
└── pyproject.toml
```

## FastAPI App Setup

### main.py Pattern

```python
"""Main FastAPI application."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from hr_chatbot_config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Import routers
from .routers import chat, setup


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events.
    
    Startup: Initialize resources
    Shutdown: Cleanup resources
    """
    # Startup
    logger.info("Starting HR Chatbot API")
    settings = get_settings()
    logger.info(f"Environment: {settings.environment}")
    
    # Initialize database
    from rag_pipeline import init_database
    init_database()
    
    yield
    
    # Shutdown
    logger.info("Shutting down HR Chatbot API")


# Create FastAPI app
app = FastAPI(
    title="HR Chatbot API",
    description="Multi-agent RAG system for HR queries",
    version="0.1.0",
    lifespan=lifespan
)

# Include routers
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(setup.router, prefix="/api/v1", tags=["setup"])


@app.get("/health", status_code=status.HTTP_200_OK, tags=["health"])
async def health_check():
    """Health check endpoint.
    
    Returns service health status and dependency checks.
    """
    from rag_pipeline import RAGPipeline
    
    checks = {
        "database": "unknown",
        "vector_store": "unknown",
        "llm": "unknown"
    }
    
    try:
        # Check database
        from sqlalchemy import create_engine, text
        settings = get_settings()
        engine = create_engine(settings.database.connection_string)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = "healthy"
    except Exception as e:
        logger.error(f"Database check failed: {e}")
        checks["database"] = "unhealthy"
    
    try:
        # Check vector store
        rag = RAGPipeline()
        # Simple check - could be improved
        checks["vector_store"] = "healthy"
    except Exception as e:
        logger.error(f"Vector store check failed: {e}")
        checks["vector_store"] = "unhealthy"
    
    # Check LLM (assumed healthy if config exists)
    checks["llm"] = "healthy"
    
    # Overall status
    overall_status = "healthy" if all(v == "healthy" for v in checks.values()) else "unhealthy"
    
    status_code = status.HTTP_200_OK if overall_status == "healthy" else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(
        status_code=status_code,
        content={
            "status": overall_status,
            "service": "api",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks
        }
    )


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "HR Chatbot API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }
```

## Request/Response Models

### models.py Pattern

```python
"""Request and response models for API endpoints."""

from pydantic import BaseModel, Field


# Chat endpoint models
class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User question or message",
        examples=["What is the vacation policy?"]
    )
    session_id: str | None = Field(
        None,
        description="Optional session ID for conversation tracking"
    )


class SourceMetadata(BaseModel):
    """Source document metadata."""
    
    title: str = Field(..., description="Document title")
    url: str = Field(..., description="Link to source document")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    
    response: str = Field(..., description="AI-generated answer")
    sources: list[SourceMetadata] = Field(
        default_factory=list,
        description="Source documents used"
    )
    session_id: str = Field(..., description="Session ID")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Answer confidence score"
    )
    escalated: bool = Field(
        default=False,
        description="Whether question was escalated to HR"
    )


# Setup endpoint models
class SetupRequest(BaseModel):
    """Request model for setup endpoint."""
    
    space_key: str | None = Field(
        None,
        description="Confluence space key (defaults to config)"
    )
    force: bool = Field(
        default=False,
        description="Force reprocessing of all pages"
    )


class SetupResponse(BaseModel):
    """Response model for setup endpoint."""
    
    status: str = Field(..., description="Setup status: completed, failed, in_progress")
    space_key: str = Field(..., description="Processed space key")
    pages_processed: int = Field(..., description="Number of pages processed")
    pages_skipped: int = Field(..., description="Number of pages skipped")
    pages_failed: int = Field(..., description="Number of pages that failed")
    duration_seconds: float = Field(..., description="Total processing time")
    chunks_created: int = Field(..., description="Total chunks created")


# Error response model
class ErrorResponse(BaseModel):
    """Standard error response."""
    
    detail: str = Field(..., description="Error message")
```

**Key Points:**
- Use Pydantic `BaseModel` for all request/response models
- Use `Field()` for validation and documentation
- Set `min_length`, `max_length`, `ge`, `le` for validation
- Provide `description` and `examples` for OpenAPI docs
- Use `| None` for optional fields with `default=None`

## Router Patterns

### chat.py Router

```python
"""Chat endpoint router."""

import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from ..models import ChatRequest, ChatResponse
from ..agents.graph import build_agent_graph, create_initial_state

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process user question with multi-agent reasoning.
    
    Workflow:
    1. Retrieve relevant documents from vector store
    2. Draft agent creates initial answer
    3. Critique agent refines answer
    4. If confidence low, escalate to HR
    
    Args:
        request: Chat request with user message
        
    Returns:
        Chat response with answer and sources
        
    Raises:
        HTTPException 400: Invalid request
        HTTPException 500: Internal processing error
        HTTPException 503: Service unavailable (dependencies down)
    """
    logger.info(f"Chat request received: {request.message[:50]}...")
    
    # Generate or use provided session ID
    session_id = request.session_id or str(uuid4())
    
    try:
        # Build agent graph
        graph = build_agent_graph()
        
        # Create initial state
        initial_state = create_initial_state(request.message)
        
        # Invoke agent workflow
        final_state = graph.invoke(initial_state)
        
        # Build response
        response = ChatResponse(
            response=final_state["final_answer"],
            sources=final_state.get("sources", []),
            session_id=session_id,
            confidence=final_state.get("confidence", 0.0),
            escalated=final_state.get("escalated", False)
        )
        
        logger.info(
            f"Chat completed: confidence={response.confidence:.2f}, "
            f"escalated={response.escalated}"
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
```

### setup.py Router

```python
"""Setup endpoint router."""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, status

from ..models import SetupRequest, SetupResponse
from rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/setup", response_model=SetupResponse, status_code=status.HTTP_200_OK)
async def setup(request: SetupRequest) -> SetupResponse:
    """Initial RAG pipeline setup - ingest all pages from Confluence space.
    
    This endpoint should typically be run once after deployment.
    Can be re-run with force=True to reprocess all pages.
    
    Args:
        request: Setup request with optional space_key and force flag
        
    Returns:
        Setup summary with processing metrics
        
    Raises:
        HTTPException 400: Invalid space key
        HTTPException 500: Setup failed
    """
    logger.info(f"Setup request received: space={request.space_key}, force={request.force}")
    
    start_time = datetime.utcnow()
    
    try:
        # Initialize RAG pipeline
        rag = RAGPipeline()
        
        # Sync space
        result = rag.sync_space(
            space_key=request.space_key,
            force=request.force
        )
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Calculate chunks created (estimate)
        chunks_created = result.get("processed", 0) * 10  # Rough estimate
        
        response = SetupResponse(
            status="completed",
            space_key=request.space_key or "default",
            pages_processed=result.get("processed", 0),
            pages_skipped=result.get("skipped", 0),
            pages_failed=result.get("failed", 0),
            duration_seconds=duration,
            chunks_created=chunks_created
        )
        
        logger.info(
            f"Setup completed: {response.pages_processed} processed, "
            f"{response.pages_skipped} skipped, {response.pages_failed} failed"
        )
        
        return response
    
    except ValueError as e:
        # Invalid space key
        logger.warning(f"Invalid setup request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        # Setup failed
        logger.error(f"Setup failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Setup failed: {str(e)}"
        )
```

## Dependency Injection Patterns

### dependencies.py

```python
"""Shared dependencies for FastAPI endpoints."""

import logging
from functools import lru_cache

from fastapi import Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from hr_chatbot_config import get_settings
from rag_pipeline import RAGPipeline
from .agents.graph import build_agent_graph

logger = logging.getLogger(__name__)


@lru_cache()
def get_rag_pipeline() -> RAGPipeline:
    """Get RAG pipeline singleton.
    
    Cached to reuse same instance across requests.
    
    Returns:
        RAG pipeline instance
    """
    logger.debug("Getting RAG pipeline instance")
    return RAGPipeline()


@lru_cache()
def get_agent_graph():
    """Get compiled agent graph singleton.
    
    Cached to avoid recompiling graph for each request.
    
    Returns:
        Compiled LangGraph
    """
    logger.debug("Getting agent graph instance")
    return build_agent_graph()


def get_db() -> Session:
    """Database session dependency.
    
    Creates session that auto-closes after request.
    
    Yields:
        SQLAlchemy session
    """
    settings = get_settings()
    engine = create_engine(settings.database.connection_string)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    try:
        yield db
    finally:
        db.close()


# Usage in endpoints
@router.get("/example")
async def example(
    rag: RAGPipeline = Depends(get_rag_pipeline),
    graph = Depends(get_agent_graph),
    db: Session = Depends(get_db)
):
    """Example endpoint with dependencies."""
    # Use rag, graph, db
    pass
```

**Key Points:**
- Use `@lru_cache()` for expensive singletons (RAG pipeline, agent graph)
- Use generator pattern (`yield`) for resources that need cleanup (DB sessions)
- Don't cache database sessions (each request gets new session)
- Dependencies are automatically injected by FastAPI

## Error Handling Patterns

### Standard Error Responses

```python
from fastapi import HTTPException, status

# 400 Bad Request - Invalid input
raise HTTPException(
    status_code=status.HTTP_400_BAD_REQUEST,
    detail="Invalid request: message cannot be empty"
)

# 404 Not Found - Resource doesn't exist
raise HTTPException(
    status_code=status.HTTP_404_NOT_FOUND,
    detail=f"Page {page_id} not found"
)

# 500 Internal Server Error - Server error
raise HTTPException(
    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    detail="Failed to process request"
)

# 503 Service Unavailable - Dependency down
raise HTTPException(
    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    detail="Vector store unavailable"
)
```

### Global Exception Handler

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler for unhandled errors.
    
    Logs error and returns generic 500 response.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )
```

### Custom Exception Handler

```python
class ServiceUnavailableError(Exception):
    """Custom exception for service unavailability."""
    pass


@app.exception_handler(ServiceUnavailableError)
async def service_unavailable_handler(request: Request, exc: ServiceUnavailableError):
    """Handle service unavailable errors."""
    logger.error(f"Service unavailable: {exc}")
    
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)}
    )
```

## Middleware Patterns

### CORS Middleware (if needed)

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Logging Middleware

```python
from fastapi import Request
import time

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with timing."""
    start_time = time.time()
    
    logger.info(f"Request: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    logger.info(
        f"Response: {request.method} {request.url.path} "
        f"status={response.status_code} duration={duration:.2f}s"
    )
    
    return response
```

### Request ID Middleware

```python
from uuid import uuid4

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to each request."""
    request_id = str(uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response
```

## Async Patterns

### Async Endpoint

```python
@router.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Async endpoint for chat."""
    # Async operations
    result = await process_async(request)
    return result
```

### Blocking Operations in Async

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

@router.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Run blocking operations in thread pool."""
    loop = asyncio.get_event_loop()
    
    # Run blocking function in executor
    result = await loop.run_in_executor(
        executor,
        blocking_function,
        request.message
    )
    
    return result
```

## Validation Patterns

### Custom Validator

```python
from pydantic import BaseModel, Field, validator

class ChatRequest(BaseModel):
    message: str = Field(...)
    
    @validator('message')
    def message_not_empty(cls, v):
        """Validate message is not empty or whitespace."""
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()
```

### Query Parameters

```python
from fastapi import Query

@router.get("/search")
async def search(
    query: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Search with validated query parameters."""
    pass
```

### Path Parameters

```python
from fastapi import Path

@router.get("/pages/{page_id}")
async def get_page(
    page_id: str = Path(..., min_length=1, max_length=50)
):
    """Get page by ID with validated path parameter."""
    pass
```

## Testing Patterns

### Test Client

```python
from fastapi.testclient import TestClient
from .main import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ["healthy", "unhealthy"]


def test_chat_endpoint():
    """Test chat endpoint."""
    response = client.post(
        "/api/v1/chat",
        json={"message": "What is the vacation policy?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "confidence" in data
```

### Test with Dependency Override

```python
from fastapi.testclient import TestClient

def mock_rag_pipeline():
    """Mock RAG pipeline for testing."""
    class MockRAG:
        def retrieve(self, query, top_k, score_threshold):
            return [{"text": "test", "title": "test", "score": 0.9}]
    return MockRAG()

app.dependency_overrides[get_rag_pipeline] = mock_rag_pipeline

client = TestClient(app)

def test_chat_with_mock():
    """Test chat with mocked dependencies."""
    response = client.post(
        "/api/v1/chat",
        json={"message": "test"}
    )
    assert response.status_code == 200
```

## Best Practices

### ✅ Do This

```python
# 1. Use type hints everywhere
@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    pass

# 2. Use status codes from status module
from fastapi import status

@router.post("/chat", status_code=status.HTTP_200_OK)
async def chat(...):
    pass

# 3. Log all endpoints
logger.info(f"Chat request: {request.message[:50]}...")

# 4. Use Pydantic for validation
class Request(BaseModel):
    field: str = Field(..., min_length=1)

# 5. Use dependency injection
async def endpoint(rag: RAGPipeline = Depends(get_rag_pipeline)):
    pass

# 6. Handle errors with HTTPException
raise HTTPException(status_code=400, detail="Invalid request")

# 7. Use response_model for type safety
@router.post("/chat", response_model=ChatResponse)

# 8. Document endpoints
async def chat(request: ChatRequest):
    """Process user question with multi-agent reasoning."""
    pass
```

### ❌ Don't Do This

```python
# 1. Don't use raw dict for responses
@router.post("/chat")
def chat(request):
    return {"response": "..."}  # Use Pydantic model

# 2. Don't use generic status codes
@router.post("/chat", status_code=200)  # Use status.HTTP_200_OK

# 3. Don't skip logging
def chat(request):
    # No logging
    pass

# 4. Don't skip validation
def chat(message: str):  # No Pydantic validation
    pass

# 5. Don't create instances in endpoints
def chat():
    rag = RAGPipeline()  # Use dependency injection
    pass

# 6. Don't return raw exceptions
def chat():
    raise Exception("Error")  # Use HTTPException

# 7. Don't skip type hints
def chat(request):  # Missing types
    pass

# 8. Don't block async endpoints
async def chat():
    time.sleep(10)  # Blocks event loop
```

## Common Patterns Summary

| Pattern | Use Case | Implementation |
|---------|----------|----------------|
| Basic Endpoint | Simple CRUD | `@router.get/post/put/delete` |
| Request Validation | Input validation | Pydantic models with `Field()` |
| Response Model | Type-safe responses | `response_model=Model` |
| Dependency Injection | Shared resources | `Depends(get_resource)` |
| Error Handling | Structured errors | `HTTPException` with status codes |
| Async Endpoint | IO-bound operations | `async def` with `await` |
| Middleware | Cross-cutting concerns | `@app.middleware("http")` |
| Health Check | Service monitoring | `/health` endpoint with checks |

## Performance Tips

1. **Cache singletons**: Use `@lru_cache()` for RAG pipeline, agent graph
2. **Use async**: For IO-bound operations (DB, API calls)
3. **Thread pool**: For CPU-bound operations in async context
4. **Connection pooling**: SQLAlchemy handles this automatically
5. **Limit request size**: Use `max_length` in Field validators
6. **Timeout handling**: Set timeouts for LLM calls