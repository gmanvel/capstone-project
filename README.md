# HR Chatbot

A multi-agent RAG (Retrieval-Augmented Generation) system that answers HR-related questions using your company's Confluence knowledge base as the source of truth.

## What It Does

The HR Chatbot enables employees to get instant, accurate answers to HR questions by:

1. **Ingesting** HR documentation from Confluence pages
2. **Understanding** questions using semantic search
3. **Reasoning** through a multi-agent workflow (draft → critique → refine)
4. **Escalating** low-confidence questions to the HR team automatically

When a user asks "What is the vacation policy?", the system retrieves relevant Confluence pages, generates an answer, self-critiques for accuracy, and returns the response with source citations.

## System Architecture

```
┌─────────────┐     ┌────────────────────────────────────────────────┐
│    User     │────▶│              API Service (FastAPI)             │
└─────────────┘     │  ┌─────────────────────────────────────────┐   │
                    │  │         LangGraph Multi-Agent           │   │
                    │  │                                         │   │
                    │  │  [Retrieve] → [Draft] → [Critique]      │   │
                    │  │                            │            │   │
                    │  │              confidence ≥ 0.6? ─────────┼───┼──▶ Response
                    │  │                    │                    │   │
                    │  │                    ▼                    │   │
                    │  │              [Notify HR]                │   │
                    │  └─────────────────────────────────────────┘   │
                    └──────────────────────┬─────────────────────────┘
                                           │
              ┌────────────────────────────┼────────────────────────────┐
              │                            │                            │
              ▼                            ▼                            ▼
     ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
     │   PostgreSQL    │        │     Qdrant      │        │  Confluence     │
     │   (metadata)    │        │   (vectors)     │        │     API         │
     └─────────────────┘        └─────────────────┘        └─────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                    Background Job (Daily Sync at Midnight)                   │
│         Detects new/updated Confluence pages → Re-indexes automatically      │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. API Service (`api/`)

FastAPI server that handles user queries through a multi-agent reasoning workflow.

**Endpoints:**
- `POST /api/v1/chat` - Ask HR questions
- `POST /api/v1/setup` - Initial knowledge base ingestion
- `GET /health` - Service health check

**Multi-Agent Workflow (LangGraph):**

| Agent | Purpose |
|-------|---------|
| **Retrieve** | Semantic search over Qdrant to find relevant document chunks |
| **Draft** | Generates initial answer from retrieved context |
| **Critique** | Reviews and refines the draft for accuracy and completeness |
| **Notify HR** | Escalates low-confidence questions (< 0.6) to HR team |

### 2. Background Job (`background-job/`)

Scheduled Python script that runs daily at midnight to keep the knowledge base synchronized.

**Behavior:**
- Fetches all pages from configured Confluence space
- Compares timestamps and content hashes to detect changes
- Re-indexes only modified pages (idempotent operation)
- Skips unchanged pages for efficiency

### 3. RAG Pipeline (`shared/rag-pipeline/`)

Shared Python module used by both services for document processing and retrieval.

**Processing Pipeline:**
1. **Fetch** - Retrieves page content from Confluence API
2. **Extract** - Parses HTML content using BeautifulSoup
3. **Chunk** - Splits text using semantic chunking (LangChain SemanticChunker)
4. **Embed** - Generates vector embeddings (Ollama locally, OpenAI in production)
5. **Store** - Saves vectors to Qdrant and metadata to PostgreSQL

**Retrieval:**
- Cosine similarity search over document embeddings
- Returns top-k chunks with metadata (title, URL) for source attribution
- Configurable score threshold for relevance filtering

### 4. Configuration (`shared/config/`)

Centralized Pydantic Settings for all services.

**Key Configuration Sections:**
- Database (PostgreSQL connection)
- Vector Store (Qdrant host/port/collection)
- Confluence (API token, base URL, space key)
- LLM (environment-aware: Ollama for dev, OpenAI for prod)
- Memory (conversation history and summarization settings)

## Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.11+ |
| Package Manager | UV (workspace mode) |
| API Framework | FastAPI |
| Agent Orchestration | LangChain + LangGraph |
| Vector Database | Qdrant |
| Relational Database | PostgreSQL |
| LLM (Development) | Ollama (Mistral/Llama) |
| LLM (Production) | OpenAI (GPT-4) |
| Embeddings (Dev) | nomic-embed-text (768 dim) |
| Embeddings (Prod) | text-embedding-3-small (1536 dim) |
| Containerization | Docker + Docker Compose |

## Running Locally

### Prerequisites

- Docker and Docker Compose
- Ollama installed locally (for development)
- Confluence API token
- (Optional) OpenAI API key for production mode

### Quick Start

1. **Clone and configure:**
   ```bash
   git clone <repository-url>
   cd capstone-project
   cp .env.example .env
   # Edit .env with your Confluence credentials
   ```

2. **Start Ollama (in a separate terminal):**
   ```bash
   ollama serve
   ollama pull mistral
   ollama pull nomic-embed-text
   ```

3. **Start all services:**
   ```bash
   docker-compose up -d
   ```

4. **Verify services are running:**
   ```bash
   # Check health
   curl http://localhost:8000/health

   # Expected response:
   # {"status":"healthy","checks":{"database":"healthy","vector_store":"healthy","llm":"healthy"}}
   ```

5. **Initialize the knowledge base:**
   ```bash
   curl -X POST http://localhost:8000/api/v1/setup \
     -H "Content-Type: application/json" \
     -d '{"space_key": "HR"}'
   ```

6. **Ask a question:**
   ```bash
   curl -X POST http://localhost:8000/api/v1/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "What is the vacation policy?"}'
   ```

### Environment Variables

Key variables in `.env`:

```bash
# Environment
ENVIRONMENT=development

# Confluence (required)
CONFLUENCE_BASE_URL=https://your-company.atlassian.net/wiki
CONFLUENCE_TOKEN=your_api_token
CONFLUENCE_SPACE_KEY=HR

# LLM Provider
LLM_PROVIDER=openai          # or "ollama" for local development
OPENAI_API_KEY=sk-...        # required if LLM_PROVIDER=openai

# Ollama (for local development)
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=mistral
```

See `.env.example` for the complete list of configuration options.

### Docker Services

| Service | Port | Description |
|---------|------|-------------|
| api | 8000 | FastAPI server |
| postgres | 5432 | Metadata storage |
| qdrant | 6333 | Vector database |
| background-job | - | Daily sync (scheduled) |

### Useful Commands

```bash
# View logs
docker-compose logs -f api

# Restart a service
docker-compose restart api

# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v

# Run evaluation suite (optional)
docker-compose --profile eval up evaluation
```

## API Usage Examples

### Chat Endpoint

```bash
# Simple question
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How many vacation days do I get?"}'

# With session ID (for conversation continuity)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What about sick leave?",
    "session_id": "user-123-session"
  }'
```

**Response:**
```json
{
  "response": "According to company policy, full-time employees receive 15 days of paid vacation annually...",
  "sources": [
    {"title": "Vacation Policy", "url": "https://confluence.company.com/..."}
  ],
  "confidence": 0.87,
  "escalated": false,
  "session_id": "user-123-session"
}
```

### Setup Endpoint

```bash
# Initial ingestion
curl -X POST http://localhost:8000/api/v1/setup \
  -H "Content-Type: application/json" \
  -d '{"space_key": "HR"}'

# Force re-index all pages
curl -X POST http://localhost:8000/api/v1/setup \
  -H "Content-Type: application/json" \
  -d '{"space_key": "HR", "force": true}'
```

**Response:**
```json
{
  "status": "completed",
  "space_key": "HR",
  "pages_processed": 47,
  "pages_skipped": 3,
  "pages_failed": 0,
  "duration_seconds": 125.4,
  "chunks_created": 485
}
```

---

## Development

### Project Structure

```
capstone-project/
├── api/                      # FastAPI service
│   └── src/hr_chatbot_api/
│       ├── agents/           # LangGraph workflow
│       ├── routers/          # API endpoints
│       └── services/         # Memory service
├── background-job/           # Scheduled sync service
├── shared/
│   ├── config/               # Pydantic Settings
│   └── rag-pipeline/         # Document processing & retrieval
├── evaluation/               # Test datasets and metrics
├── architecture/             # Design documentation
├── docker-compose.yml
└── .env.example
```

### Running Tests

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run specific test module
uv run pytest shared/rag-pipeline/tests/
```

### Code Style

- Type hints required on all functions
- Google-style docstrings
- Use `get_settings()` for configuration (never `os.getenv()`)
- LangGraph nodes return new dicts (immutable state)

See `architecture/CODING_STANDARDS.md` for detailed guidelines.
