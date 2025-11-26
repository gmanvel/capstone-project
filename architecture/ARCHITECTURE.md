# HR Chatbot - System Architecture

## System Overview

Multi-agent RAG system for HR queries using Confluence as knowledge base. The system consists of two deployable units that share common RAG pipeline logic.

## Architecture Principles

1. **Simplicity**: Two deployable units sharing a common RAG pipeline module
2. **Environment Parity**: Local (Ollama) must match production (OpenAI) behavior
3. **Idempotency**: All data operations are safe to retry
4. **Shared Code**: RAG pipeline is an importable Python module, not a microservice

## Deployable Units

### 1. API Service (FastAPI)

**Purpose**: User-facing chatbot API with multi-agent reasoning

**Runs**: Continuous (uvicorn server)

**Port**: 8000

**Dependencies**: 
- Postgres (metadata storage)
- Qdrant (vector search)
- Confluence API (for initial setup)
- LLM (Ollama for dev, OpenAI for prod)

**Endpoints**:
- `POST /chat` - Question answering with draft → critique agents
- `POST /setup` - Initial RAG pipeline setup (one-time ingestion)
- `GET /health` - Health check

**Responsibilities**:
- Receive user questions
- Orchestrate multi-agent workflow (draft → critique)
- Call RAG pipeline for document retrieval
- Handle low confidence with notify_hr tool
- Return answers with source attribution

### 2. Background Job (Scheduled Python Script)

**Purpose**: Daily Confluence synchronization

**Runs**: Daily at midnight (cron/scheduler)

**Dependencies**:
- Postgres (metadata storage)
- Qdrant (vector search)
- Confluence API (fetch pages)
- LLM (Ollama for dev, OpenAI for prod)

**Behavior**:
- Fetch all pages from Confluence space
- For each page:
  - If page doesn't exist in DB → ingest
  - If page exists but outdated (compare last_updated) → re-ingest
  - If page up-to-date → skip

**Responsibilities**:
- Detect new Confluence pages
- Detect updated Confluence pages
- Call RAG pipeline to process changed pages
- Log sync summary

## Shared Components

### RAG Pipeline Module

**Location**: `shared/rag-pipeline/`

**Type**: Importable Python package (NOT a microservice)

**Used By**: Both API service and Background Job

**Core Responsibilities**:

1. **Ingestion**: Fetch page content from Confluence API
2. **Chunking**: Split text using semantic chunking
   - use SemanticChunker from langchain_experimental
   - make semantic_breakpoint_threshold configurable
3. **Embedding**: Generate embeddings using environment-aware model
   - Development: Ollama (nomic-embed-text)
   - Production: OpenAI (text-embedding-ada-002)
4. **Storage**: Store metadata + vectors
   - Metadata → Postgres (confluence_pages table)
   - Vectors → Qdrant (hr_documents collection)
5. **Retrieval**: Semantic search from Qdrant

**Key Features**:
- Environment-aware (automatically switches between Ollama/OpenAI)
- Idempotent (checks before processing using page hash)
- Transaction-safe (rollback on failure)
- Provides public API: retrieve(), process_page(), sync_space(), should_process_page()

### Configuration Module

**Location**: `shared/config/`

**Type**: Pydantic Settings

**Purpose**: Centralized configuration management for all services

**Provides**:
- DatabaseConfig (Postgres connection)
- QdrantConfig (vector store)
- LLMConfig (environment-aware Ollama/OpenAI)
- ConfluenceConfig (API token, base_url, space_key)

## Data Flow

### 1. Initial Setup Flow (POST /setup)
```
User → API: POST /setup {"space_key": "HR"}
  ↓
API → Confluence API: Get all pages in space
  ↓
For each page:
  API → RAG Pipeline: process_page(page_id)
    ↓
  RAG Pipeline → Postgres: Check if page exists & check content_hash
    ↓
  If needs processing:
    ↓
    RAG Pipeline → Confluence API: Fetch full page content
    ↓
    RAG Pipeline: Extract text from HTML (BeautifulSoup)
    ↓
    RAG Pipeline: Compute SHA256 hash of content
    ↓
    RAG Pipeline: Split into chunks (1000 chars, 200 overlap)
    ↓
    RAG Pipeline → LLM: Generate embeddings for each chunk
    ↓
    RAG Pipeline → Postgres: Store page metadata
      (id, title, space_key, content_hash, last_updated, version, url, synced_at)
    ↓
    RAG Pipeline → Qdrant: Store vectors with payload
      (page_id, title, chunk_index, text, space_key, url, last_modified)
  ↓
  If already processed & up-to-date:
    ↓
    RAG Pipeline: Skip (log "already up-to-date")
  ↓
API → User: Setup complete
  {
    "status": "completed",
    "pages_processed": 47,
    "pages_skipped": 3,
    "chunks_created": 485
  }
```

### 2. Question Answering Flow (POST /chat)
```
User → API: POST /chat {"message": "What is the vacation policy?"}
  ↓
API → LangGraph: Start agent workflow
  ↓
LangGraph: retrieve_node
  ↓
  retrieve_node → RAG Pipeline: retrieve(query, top_k=5, score_threshold=0.7)
    ↓
  RAG Pipeline → LLM: Generate query embedding
    ↓
  RAG Pipeline → Qdrant: Similarity search (cosine distance)
    ↓
  RAG Pipeline → retrieve_node: Return top 5 chunks with metadata
  ↓
LangGraph: draft_node
  ↓
  draft_node: Build context from retrieved chunks
  ↓
  draft_node → LLM: Generate draft answer
    SystemMessage: "You are an HR assistant. Answer based on provided context."
    HumanMessage: "Context: {chunks}\n\nQuestion: {query}"
  ↓
  draft_node: Parse response, estimate confidence
  ↓
  draft_node → State: Update {draft_answer, confidence}
  ↓
LangGraph: critique_node
  ↓
  critique_node → LLM: Critique and refine draft
    SystemMessage: "Review and improve the draft answer."
    HumanMessage: "Draft: {draft}\n\nContext: {chunks}\n\nQuestion: {query}"
  ↓
  critique_node → State: Update {critique, final_answer, confidence}
  ↓
LangGraph: confidence_check (conditional edge)
  ↓
  If confidence >= 0.7:
    ↓
    → END: Return final_answer + sources
  ↓
  If confidence < 0.7:
    ↓
    LangGraph: notify_hr_node
      ↓
      notify_hr_node: Log question for HR review
      notify_hr_node: (Optional) Send email/Slack notification
      ↓
      notify_hr_node → State: Update {final_answer: "escalated message", escalated: true}
      ↓
    → END: Return escalation message
  ↓
API → User: Return response
  {
    "response": "Based on company policy...",
    "sources": [{"title": "...", "url": "..."}],
    "confidence": 0.89,
    "escalated": false
  }
```

### 3. Background Sync Flow (Daily at Midnight)
```
Cron/Scheduler → Background Job: Execute at midnight
  ↓
Background Job → Confluence API: Get all pages in space
  ↓
For each page:
  ↓
  Background Job → RAG Pipeline: sync_page(page_id, last_modified)
    ↓
  RAG Pipeline → Postgres: Query confluence_pages WHERE id = page_id
    ↓
  Case 1: Page NOT in database
    ↓
    RAG Pipeline: Process page (full ingestion flow)
      → Fetch content
      → Chunk text
      → Generate embeddings
      → Store in Postgres + Qdrant
    ↓
  Case 2: Page exists in database
    ↓
    RAG Pipeline: Compare Confluence.last_modified vs DB.last_updated
      ↓
    If Confluence.last_modified > DB.last_updated:
      ↓
      RAG Pipeline → Qdrant: Delete old embeddings
        (delete where page_id = {page_id})
      ↓
      RAG Pipeline: Process page (full ingestion flow)
        → Fetch content
        → Chunk text
        → Generate embeddings
        → Update Postgres record
        → Store new vectors in Qdrant
    ↓
    If Confluence.last_modified <= DB.last_updated:
      ↓
      RAG Pipeline: Skip (already up-to-date)
  ↓
Background Job: Log sync summary
  {
    "total_pages": 50,
    "processed": 5,
    "skipped": 44,
    "failed": 1,
    "duration_seconds": 45.2
  }
  ↓
Background Job: Exit
```

## Data Models

### Postgres Schema

**Table: confluence_pages**

Tracks synchronized Confluence pages for idempotency and change detection.
```sql
CREATE TABLE confluence_pages (
    id VARCHAR PRIMARY KEY,              -- Confluence page ID (e.g., "123456")
    title VARCHAR NOT NULL,              -- Page title
    space_key VARCHAR NOT NULL,          -- Confluence space key (e.g., "HR")
    content_hash VARCHAR NOT NULL,       -- SHA256 hash of content (for change detection)
    last_updated TIMESTAMPTZ NOT NULL,   -- From Confluence 'when' metadata (page last modified time, stored in UTC)
    version INTEGER NOT NULL,            -- Confluence page version number
    url VARCHAR NOT NULL,                -- Full URL to page
    synced_at TIMESTAMPTZ DEFAULT NOW()  -- When we last synced this page (stored in UTC)
);

-- Indexes for performance
CREATE INDEX idx_space_key ON confluence_pages(space_key);
CREATE INDEX idx_last_updated ON confluence_pages(last_updated);
```

**Purpose**:
- Track which pages have been processed
- Detect changes by comparing content_hash
- Detect updates by comparing last_updated timestamps
- Provide audit trail with synced_at

### Qdrant Schema

**Collection: hr_documents**

Stores document embeddings with metadata for semantic search.

**Configuration**:
```python
{
    "collection_name": "hr_documents",
    "vectors_config": {
        "size": 768,      # nomic-embed-text (dev) OR 1536 for text-embedding-3-small (prod)
        "distance": "Cosine"
    }
}
```

**Point Structure**:
```python
{
    "id": "123456_0",  # Format: {page_id}_{chunk_index}
    "vector": [0.1, 0.2, ..., 0.5],  # Embedding vector (768 or 1536 dimensions)
    "payload": {
        "page_id": "123456",
        "title": "Vacation Policy",
        "chunk_index": 0,
        "text": "Employees are entitled to 15 days of paid vacation annually...",
        "space_key": "HR",
        "url": "https://confluence.company.com/pages/viewpage.action?pageId=123456",
        "last_modified": "2025-01-15T10:30:00Z"
    }
}
```

**Purpose**:
- Store vector embeddings for semantic search
- Include full text in payload for context retrieval
- Include metadata for source attribution
- Enable filtering by space_key, page_id

## LangGraph Agent Architecture

### State Definition
```python
from typing import TypedDict, Any

class AgentState(TypedDict):
    """State shared across all agent nodes.
    
    All fields must be typed. State is immutable - nodes return new dicts.
    """
    query: str                          # User's original question
    retrieved_docs: list[dict[str, Any]]  # Documents from RAG pipeline
    draft_answer: str | None            # Initial answer from Drafter
    critique: str | None                # Feedback from Critic
    final_answer: str                   # Final refined answer
    confidence: float                   # Confidence score (0.0 to 1.0)
    sources: list[dict[str, str]]       # Source metadata for user (title + url)
```

### Agent Nodes

#### 1. retrieve_node

**Input**: `query`

**Output**: Updates `retrieved_docs`

**Behavior**:
```python
def retrieve_node(state: AgentState) -> dict[str, Any]:
    """Retrieve relevant documents from RAG pipeline."""
    rag_pipeline = RAGPipeline()
    docs = rag_pipeline.retrieve(
        query=state["query"],
        top_k=5,
        score_threshold=0.7
    )
    return {"retrieved_docs": docs}
```

#### 2. draft_node

**Input**: `query`, `retrieved_docs`

**Output**: Updates `draft_answer`, `confidence`

**Behavior**:
```python
def draft_node(state: AgentState) -> dict[str, Any]:
    """Create initial answer from retrieved context."""
    # Build context from retrieved docs
    context = "\n\n".join([
        f"Source: {doc['title']}\n{doc['text']}"
        for doc in state["retrieved_docs"]
    ])
    
    # Prompt LLM
    messages = [
        SystemMessage(content="You are an HR assistant. Answer based only on provided context."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['query']}")
    ]
    
    response = llm.invoke(messages)
    
    # Estimate confidence (could be improved with LLM self-assessment)
    confidence = 0.85 if len(state["retrieved_docs"]) >= 3 else 0.6
    
    return {
        "draft_answer": response.content,
        "confidence": confidence
    }
```

#### 3. critique_node

**Input**: `query`, `retrieved_docs`, `draft_answer`

**Output**: Updates `critique`, `final_answer`, `confidence`

**Behavior**:
```python
def critique_node(state: AgentState) -> dict[str, Any]:
    """Critique and refine the draft answer."""
    context = "\n\n".join([doc["text"] for doc in state["retrieved_docs"]])
    
    messages = [
        SystemMessage(content="Review the draft answer. Improve accuracy, clarity, and completeness."),
        HumanMessage(content=f"""
Draft Answer: {state['draft_answer']}

Context: {context}

Original Question: {state['query']}

Provide a refined answer.
""")
    ]
    
    response = llm.invoke(messages)
    
    # Potentially adjust confidence based on critique
    confidence = state["confidence"]
    
    # Extract sources
    sources = [
        {"title": doc["title"], "url": doc["url"]}
        for doc in state["retrieved_docs"]
    ]
    
    return {
        "critique": "Applied improvements",
        "final_answer": response.content,
        "confidence": confidence,
        "sources": sources
    }
```

#### 4. notify_hr_node (Conditional)

**Input**: `query`, `confidence`

**Output**: Updates `final_answer`, `escalated` flag

**Behavior**:
```python
def notify_hr_node(state: AgentState) -> dict[str, Any]:
    """Escalate low-confidence questions to HR."""
    # Log for HR review
    logger.warning(f"Low confidence question escalated: {state['query']}")
    
    # TODO: Send email/Slack notification to HR
    # send_notification(state["query"])
    
    return {
        "final_answer": "I couldn't find a confident answer to your question. "
                       "Your question has been forwarded to the HR team, and they will respond shortly.",
        "escalated": True,
        "sources": []
    }
```

### Conditional Edge
```python
def should_notify_hr(state: AgentState) -> str:
    """Determine if question should be escalated to HR."""
    CONFIDENCE_THRESHOLD = 0.7
    
    if state["confidence"] < CONFIDENCE_THRESHOLD:
        return "notify"
    return "end"
```

### Graph Construction
```python
from langgraph.graph import StateGraph, END

# Create graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("draft", draft_node)
workflow.add_node("critique", critique_node)
workflow.add_node("notify_hr", notify_hr_node)

# Add edges
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "draft")
workflow.add_edge("draft", "critique")

# Conditional edge based on confidence
workflow.add_conditional_edges(
    "critique",
    should_notify_hr,
    {
        "notify": "notify_hr",
        "end": END
    }
)

workflow.add_edge("notify_hr", END)

# Compile
agent_graph = workflow.compile()
```

## Technology Stack

### Core Technologies
- **Language**: Python 3.11+
- **Package Manager**: UV (workspace mode)
- **API Framework**: FastAPI
- **LLM Orchestration**: LangChain + LangGraph
- **Vector Database**: Qdrant
- **SQL Database**: PostgreSQL
- **Containerization**: Docker + Docker Compose

### Key Dependencies

**Shared Config**:
- pydantic>=2.0.0
- pydantic-settings>=2.0.0

**Shared RAG Pipeline**:
- langchain>=0.1.0
- langchain-community>=0.0.20
- langchain-openai>=0.0.5
- qdrant-client>=1.7.0
- sqlalchemy>=2.0.25
- psycopg2-binary>=2.9.9
- atlassian-python-api>=3.41.0
- beautifulsoup4>=4.12.0

**API Service**:
- fastapi>=0.109.0
- uvicorn>=0.27.0
- langgraph>=0.0.20

**Background Job**:
- schedule>=1.2.0

### LLM Models

**Development (Local)**:
- Chat: Ollama llama3.1:8b
- Embeddings: Ollama nomic-embed-text (768 dimensions)
- Base URL: http://localhost:11434

**Production (OpenAI)**:
- Chat: gpt-4o-mini
- Embeddings: text-embedding-ada-002 (1536 dimensions)
- Requires: OPENAI_API_KEY

## Configuration Strategy

### Environment Variables

All configuration managed via `.env` file and loaded by Pydantic Settings.

**Required Variables**:
```bash
# Environment mode
ENVIRONMENT=development  # or "production"

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=hr_chatbot
POSTGRES_USER=hrbot
POSTGRES_PASSWORD=your_password

# Vector Store
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=hr_documents

# Confluence
CONFLUENCE_TOKEN=your_api_token
CONFLUENCE_BASE_URL=https://your-company.atlassian.net/wiki
CONFLUENCE_SPACE_KEY=HR

# LLM (Development - Ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# LLM (Production - OpenAI)
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002

# Agent Configuration
CONFIDENCE_THRESHOLD=0.7
RETRIEVAL_TOP_K=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Configuration Access Pattern

**Always use get_settings() - never os.getenv() directly**
```python
from hr_chatbot_config import get_settings

settings = get_settings()

# Access configuration
db_url = settings.database.connection_string
qdrant_url = settings.qdrant.url
is_local = settings.llm.is_local

# Environment-aware LLM selection
if settings.llm.is_local:
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings import OllamaEmbeddings
    
    llm = ChatOllama(
        base_url=settings.llm.ollama_base_url,
        model=settings.llm.chat_model_name
    )
    embeddings = OllamaEmbeddings(
        base_url=settings.llm.ollama_base_url,
        model=settings.llm.embedding_model_name
    )
else:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    
    llm = ChatOpenAI(
        api_key=settings.llm.openai_api_key,
        model=settings.llm.chat_model_name
    )
    embeddings = OpenAIEmbeddings(
        api_key=settings.llm.openai_api_key,
        model=settings.llm.embedding_model_name
    )
```

## Idempotency & Safety

### Page Processing Idempotency

**Goal**: Safe to re-run without duplicating work or corrupting data.

**Mechanism**: Hash-based change detection + timestamp comparison
```python
def should_process_page(
    page_id: str,
    confluence_last_modified: datetime,
    confluence_content: str
) -> bool:
    """Check if page needs processing.
    
    Returns:
        True if page should be processed (new or changed)
        False if page is already up-to-date
    """
    # Compute hash of new content
    new_hash = hashlib.sha256(confluence_content.encode()).hexdigest()
    
    # Check if page exists in database
    existing = session.query(ConfluencePage).filter_by(id=page_id).first()
    
    if not existing:
        return True  # New page - must process
    
    if new_hash != existing.content_hash:
        return True  # Content changed - must reprocess
    
    if confluence_last_modified > existing.last_updated:
        return True  # Timestamp newer - must reprocess
    
    return False  # Already up-to-date - skip
```

### Transaction Safety

**Goal**: All-or-nothing processing - no partial state.

**Mechanism**: Database transactions + cleanup on failure
```python
def process_page(page_id: str) -> dict[str, Any]:
    """Process page with rollback on failure."""
    session = get_db_session()
    
    try:
        # 1. Fetch content from Confluence
        page_data = confluence_client.get_page(page_id)
        text = extract_text(page_data["body"]["storage"]["value"])
        
        # 2. Chunk text
        chunks = text_splitter.split_text(text)
        
        # 3. Generate embeddings
        embeddings = embeddings_model.embed_documents(chunks)
        
        # 4. Store in Qdrant
        points = [
            PointStruct(
                id=f"{page_id}_{idx}",
                vector=embedding,
                payload={
                    "page_id": page_id,
                    "text": chunk,
                    # ... metadata
                }
            )
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        qdrant_client.upsert(collection_name="hr_documents", points=points)
        
        # 5. Store metadata in Postgres
        page_record = ConfluencePage(
            id=page_id,
            title=page_data["title"],
            content_hash=compute_hash(text),
            # ... other fields
        )
        session.add(page_record)
        session.commit()
        
        return {"status": "processed", "chunks": len(chunks)}
        
    except Exception as e:
        # Rollback database
        session.rollback()
        
        # Cleanup Qdrant (delete any partial inserts)
        try:
            qdrant_client.delete(
                collection_name="hr_documents",
                points_selector={"filter": {"must": [{"key": "page_id", "match": {"value": page_id}}]}}
            )
        except:
            pass
        
        logger.error(f"Failed to process page {page_id}: {e}", exc_info=True)
        raise
        
    finally:
        session.close()
```

## Deployment

### Docker Compose Services
```yaml
services:
  postgres:       # PostgreSQL database
  qdrant:         # Vector database
  ollama:         # Local LLM (dev only, profile=development)
  api:            # FastAPI service
  background-job: # Scheduled sync
```

### Service Dependencies
```
API depends_on:
  - postgres (healthy)
  - qdrant (healthy)

Background Job depends_on:
  - postgres (healthy)
  - qdrant (healthy)
```

### Health Checks

All services must implement health checks:

**Postgres**: `pg_isready -U hrbot`

**Qdrant**: `curl -f http://localhost:6333/health`

**Ollama**: `curl -f http://localhost:11434/api/tags`

**API**: `curl -f http://localhost:8000/health`

### Startup Order

1. Infrastructure services start (postgres, qdrant, ollama)
2. Wait for all to be healthy
3. Application services start (api, background-job)

## Security Considerations

### Secrets Management
- Never commit `.env` file
- Use environment variables for all secrets
- Confluence token stored securely
- OpenAI API key never logged