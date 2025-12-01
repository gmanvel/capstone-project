# API Contracts

Complete API specification for the HR Chatbot API service.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com`

---

## Endpoints

### 1. POST /api/v1/chat

Process user question with multi-agent reasoning workflow.

**Description**: Executes a LangGraph workflow that retrieves relevant documents, generates a draft answer, critiques and refines it, and optionally escalates low-confidence questions to HR.

**Request**

```json
{
  "message": "string (required, 1-2000 characters)",
  "session_id": "string (optional)"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | Yes | User question or message (1-2000 characters) |
| `session_id` | string | No | Optional session ID for conversation tracking |

**Response (200 OK)**

```json
{
  "response": "string",
  "sources": [
    {
      "title": "string",
      "url": "string"
    }
  ],
  "session_id": "string",
  "confidence": 0.85,
  "escalated": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `response` | string | AI-generated answer to the user's question |
| `sources` | array | List of source documents used for the answer |
| `sources[].title` | string | Document title |
| `sources[].url` | string | Link to source document |
| `session_id` | string | Session ID for conversation tracking |
| `confidence` | float | Answer confidence score (0.0 to 1.0) |
| `escalated` | boolean | Whether question was escalated to HR team |

**Error Responses**

**400 Bad Request**
```json
{
  "detail": "string"
}
```
Reasons:
- Empty message
- Message exceeds 2000 characters
- Invalid request format

**500 Internal Server Error**
```json
{
  "detail": "Failed to process chat request"
}
```

**Example**

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the vacation policy?",
    "session_id": "abc123"
  }'
```

---

### 2. POST /api/v1/setup

Initial RAG pipeline setup - ingest Confluence pages.

**Description**: Performs one-time ingestion of Confluence pages into the RAG pipeline. Fetches all pages from specified space, chunks them, generates embeddings, and stores in vector database (Qdrant) and metadata store (PostgreSQL).

**Request**

```json
{
  "space_key": "string (optional)",
  "force": false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `space_key` | string | No | From config | Confluence space key to sync |
| `force` | boolean | No | false | Force reprocessing of all pages (ignores cache) |

**Response (200 OK)**

```json
{
  "status": "completed",
  "space_key": "HR",
  "pages_processed": 42,
  "pages_skipped": 15,
  "pages_failed": 0,
  "duration_seconds": 127.5,
  "chunks_created": 856
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Setup status: "completed", "failed", or "in_progress" |
| `space_key` | string | Confluence space key that was processed |
| `pages_processed` | integer | Number of pages successfully processed |
| `pages_skipped` | integer | Number of pages skipped (unchanged since last sync) |
| `pages_failed` | integer | Number of pages that failed processing |
| `duration_seconds` | float | Total processing time in seconds |
| `chunks_created` | integer | Total number of document chunks created |

**Error Responses**

**400 Bad Request**
```json
{
  "detail": "string"
}
```
Reasons:
- Invalid space key
- Invalid parameters

**500 Internal Server Error**
```json
{
  "detail": "Setup failed: <error message>"
}
```

**Example**

```bash
curl -X POST http://localhost:8000/api/v1/setup \
  -H "Content-Type: application/json" \
  -d '{
    "space_key": "HR",
    "force": false
  }'
```

---

### 3. GET /health

Health check endpoint.

**Description**: Checks status of all service dependencies: database (PostgreSQL), vector store (Qdrant), and LLM. Returns 200 if all healthy, 503 if any component is unhealthy.

**Request**

No parameters required.

**Response (200 OK)**

```json
{
  "status": "healthy",
  "service": "api",
  "timestamp": "2024-01-15T10:30:00.000000",
  "checks": {
    "database": "healthy",
    "vector_store": "healthy",
    "llm": "healthy"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Overall status: "healthy" or "unhealthy" |
| `service` | string | Service identifier (always "api") |
| `timestamp` | string | ISO 8601 timestamp |
| `checks` | object | Individual component health status |
| `checks.database` | string | PostgreSQL status: "healthy", "unhealthy", or "unknown" |
| `checks.vector_store` | string | Qdrant status: "healthy", "unhealthy", or "unknown" |
| `checks.llm` | string | LLM status: "healthy", "unhealthy", or "unknown" |

**Response (503 Service Unavailable)**

Same structure as 200, but with `status: "unhealthy"` and one or more checks showing "unhealthy".

**Example**

```bash
curl http://localhost:8000/health
```

---

### 4. GET /

Root endpoint with API information.

**Description**: Provides basic API service information and links to documentation.

**Request**

No parameters required.

**Response (200 OK)**

```json
{
  "service": "HR Chatbot API",
  "version": "0.1.0",
  "docs": "/docs",
  "health": "/health"
}
```

**Example**

```bash
curl http://localhost:8000/
```

---

## Data Models

### ChatRequest

```python
class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User question or message"
    )
    session_id: str | None = Field(
        None,
        description="Optional session ID for conversation tracking"
    )
```

### ChatResponse

```python
class ChatResponse(BaseModel):
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
```

### SourceMetadata

```python
class SourceMetadata(BaseModel):
    title: str = Field(..., description="Document title")
    url: str = Field(..., description="Link to source document")
```

### SetupRequest

```python
class SetupRequest(BaseModel):
    space_key: str | None = Field(
        None,
        description="Confluence space key (defaults to config)"
    )
    force: bool = Field(
        default=False,
        description="Force reprocessing of all pages"
    )
```

### SetupResponse

```python
class SetupResponse(BaseModel):
    status: str = Field(..., description="Setup status: completed, failed, in_progress")
    space_key: str = Field(..., description="Processed space key")
    pages_processed: int = Field(..., description="Number of pages processed")
    pages_skipped: int = Field(..., description="Number of pages skipped")
    pages_failed: int = Field(..., description="Number of pages that failed")
    duration_seconds: float = Field(..., description="Total processing time")
    chunks_created: int = Field(..., description="Total chunks created")
```

---

## HTTP Status Codes

| Code | Status | Description |
|------|--------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid request (validation error) |
| 500 | Internal Server Error | Server-side processing error |
| 503 | Service Unavailable | Service or dependency is unhealthy |

---

## Agent Workflow

The `/api/v1/chat` endpoint executes the following multi-agent workflow:

```
User Query
    ↓
[Retrieve Memory] - Load conversation history
    ↓
[Retrieve] - Similarity search in Qdrant
    ↓
[Draft] - Generate initial answer
    ↓
[Critique] - Review and improve answer
    ↓
[Confidence Check]
    ├── confidence ≥ 0.6 → Return answer
    └── confidence < 0.6 → [Notify HR] → Return answer with escalated=true
```

---

## Authentication

Currently, the API does not require authentication. All endpoints are publicly accessible.

**Note**: For production deployment, implement API key authentication or OAuth 2.0.

---

## Rate Limiting

Currently, no rate limiting is implemented.

**Note**: For production deployment, implement rate limiting to prevent abuse.

---

## CORS

The API does not currently configure CORS headers.

**Note**: For web client integration, configure CORS middleware in FastAPI.

---

## OpenAPI Documentation

Interactive API documentation is available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

---

## Versioning

The API uses URL path versioning: `/api/v1/...`

Current version: **v1**

---

## Error Handling

All errors follow this format:

```json
{
  "detail": "Human-readable error message"
}
```

The API uses appropriate HTTP status codes and provides descriptive error messages.

### Common Error Scenarios

| Scenario | Status | Detail |
|----------|--------|--------|
| Empty message | 400 | Validation error |
| Message too long (>2000 chars) | 400 | Validation error |
| Database connection failed | 500 | Internal server error |
| Confluence API unavailable | 500 | Setup failed: ... |
| Invalid space key | 400 | Invalid space key |

---

## Notes

1. **Idempotency**: The `/setup` endpoint is idempotent when `force=false`. Pages with unchanged content hashes are skipped.

2. **Session Management**: Session IDs are generated if not provided. Sessions are stored in memory and conversation history is maintained for context.

3. **Memory**: Conversation memory is optional (configured via `MEMORY_ENABLED` environment variable). When enabled, recent messages are summarized after reaching the threshold.

4. **Confidence Threshold**: Questions with confidence < 0.6 trigger HR escalation. This threshold is configurable via `CONFIDENCE_THRESHOLD` environment variable.

5. **Environment Awareness**: The system automatically selects between Ollama (development) and OpenAI (production) based on configuration.
