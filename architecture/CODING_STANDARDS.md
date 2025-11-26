```markdown
# Coding Standards

## Core Principles

1. **Explicit over Implicit**: Prefer clarity over cleverness
2. **Type Safety**: All functions have complete type annotations
3. **Fail Fast**: Validate inputs immediately, raise exceptions early
4. **Observability**: Use structured logging, never print()
5. **Consistency**: Follow established patterns, don't introduce new styles

## Python Style Guide

### Formatting

- **Line Length**: 100 characters
- **Formatter**: Black (run before committing)
- **Linter**: Ruff (run before committing)
- **Import Sorting**: Automatic via Ruff

### Type Hints

**All functions must have complete type annotations.**

```python
# ✅ Good: Complete type annotations
def process_page(page_id: str, force: bool = False) -> dict[str, Any]:
    """Process a Confluence page."""
    pass

def retrieve(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Retrieve relevant documents."""
    pass

# ❌ Bad: Missing types
def process_page(page_id, force=False):
    pass

def retrieve(query, top_k=5):
    pass
```

**Use modern type hints (Python 3.11+)**:

```python
# ✅ Good: Modern syntax
from typing import Any

def get_config() -> dict[str, Any]:
    pass

def get_pages() -> list[str]:
    pass

def get_page(page_id: str) -> dict[str, Any] | None:
    pass

# ❌ Bad: Old syntax
from typing import Dict, List, Optional, Any

def get_config() -> Dict[str, Any]:
    pass

def get_pages() -> List[str]:
    pass

def get_page(page_id: str) -> Optional[Dict[str, Any]]:
    pass
```

### Docstrings

**All public functions must have docstrings in Google style.**

```python
def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.7) -> list[dict[str, Any]]:
    """Retrieve relevant documents using similarity search.
    
    Args:
        query: User query text
        top_k: Number of results to return (default: 5)
        score_threshold: Minimum similarity score (default: 0.7)
        
    Returns:
        List of document chunks with metadata:
        - text: Chunk content
        - page_id: Source page ID
        - title: Page title
        - score: Similarity score
        
    Raises:
        ValueError: If top_k < 1 or score_threshold out of range [0, 1]
        QdrantException: If vector store query fails
    """
    pass
```

**Private/internal functions can have brief docstrings**:

```python
def _compute_hash(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()
```

### File Organization

**Every Python file must follow this structure**:

```python
"""Module docstring explaining purpose.

This module handles Confluence page synchronization.
"""

# 1. Standard library imports
import logging
from datetime import datetime
from typing import Any

# 2. Third-party imports
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

# 3. Local imports
from hr_chatbot_config import get_settings
from rag_pipeline import RAGPipeline

# 4. Constants
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
CONFIDENCE_THRESHOLD = 0.7

# 5. Module-level logger
logger = logging.getLogger(__name__)

# 6. Classes and functions
class MyClass:
    pass

def my_function():
    pass
```

## Naming Conventions

### Project Structure

```
# Folders: kebab-case
api/
background-job/
rag-pipeline/

# Python packages: snake_case
hr_chatbot_config/
rag_pipeline/

# Docker services: kebab-case
postgres
api
background-job
```

### Code Naming

```python
# Classes: PascalCase
class ConfluencePage:
    pass

class RAGPipeline:
    pass

class AgentState:
    pass

# Functions/Methods: snake_case
def process_page(page_id: str) -> dict[str, Any]:
    pass

def get_settings() -> Settings:
    pass

async def chat(request: ChatRequest) -> ChatResponse:
    pass

# Variables: snake_case
page_id = "123456"
retrieved_docs = []
confluence_client = ConfluenceClient()

# Constants: SCREAMING_SNAKE_CASE
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
CHUNK_SIZE = 1000
CONFIDENCE_THRESHOLD = 0.7

# Private (module/class internal): _leading_underscore
def _compute_hash(content: str) -> str:
    pass

_session_factory = None
_cache = {}

# Boolean variables: Use is/has/should/can prefix
is_processed = True
has_changed = False
should_retry = True
can_process = True
```

### File Naming

```python
# Python files: snake_case
confluence_client.py
rag_pipeline.py
agent_graph.py

# Test files: test_ prefix
test_confluence_client.py
test_rag_pipeline.py
test_agent_graph.py
```

## Configuration Management

### Always Use Settings

**Never use os.getenv() directly. Always use the shared config module.**

```python
# ✅ Good: Use shared config
from hr_chatbot_config import get_settings

def __init__(self):
    self.settings = get_settings()
    self.qdrant_url = self.settings.qdrant.url
    self.chunk_size = 1000  # Or from settings if configurable
    self.confluence_token = self.settings.confluence.token

# ❌ Bad: Direct environment access
import os

def __init__(self):
    self.qdrant_url = os.getenv("QDRANT_URL")  # Don't do this
    self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))  # Don't do this
    self.confluence_token = os.getenv("CONFLUENCE_TOKEN")  # Don't do this
```

### Environment-Aware Code

**Always check environment flag when choosing LLM/embeddings.**

```python
# ✅ Good: Environment-aware
from hr_chatbot_config import get_settings

settings = get_settings()

if settings.llm.is_local:
    # Development: Ollama
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
    # Production: OpenAI
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    
    llm = ChatOpenAI(
        api_key=settings.llm.openai_api_key,
        model=settings.llm.chat_model_name
    )
    embeddings = OpenAIEmbeddings(
        api_key=settings.llm.openai_api_key,
        model=settings.llm.embedding_model_name
    )

# ❌ Bad: Hardcoded model
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")  # Won't work in production
```

## Error Handling

### Exception Patterns

**Always use specific exceptions with context and logging.**

```python
import logging

logger = logging.getLogger(__name__)

# ✅ Good: Specific exceptions with logging
def process_data(data: dict[str, Any]) -> dict[str, Any]:
    try:
        result = transform(data)
        return result
    
    except KeyError as e:
        logger.error(f"Missing required field: {e}")
        raise ValueError(f"Missing required field: {e}") from e
    
    except ValidationError as e:
        logger.warning(f"Validation failed: {e}")
        raise
    
    except Exception as e:
        logger.exception(f"Unexpected error processing data: {e}")
        raise

# ❌ Bad: Bare except, no logging
def process_data(data):
    try:
        result = transform(data)
        return result
    except:
        return None  # Silent failure
```

### No Silent Failures

```python
# ✅ Good: Log and raise
try:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
except requests.Timeout:
    logger.error(f"Request timeout: {url}")
    raise
except requests.HTTPError as e:
    logger.error(f"HTTP error {e.response.status_code}: {url}")
    raise
except requests.RequestException as e:
    logger.error(f"Request failed: {url} - {e}")
    raise

# ❌ Bad: Silent failure
try:
    response = requests.get(url)
except:
    pass  # Error information lost
```

### Input Validation

**Validate inputs at function entry. Fail fast.**

```python
# ✅ Good: Validate early
def retrieve(query: str, top_k: int = 5, score_threshold: float = 0.7) -> list[dict[str, Any]]:
    """Retrieve relevant documents."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    if top_k < 1 or top_k > 20:
        raise ValueError(f"top_k must be between 1 and 20, got {top_k}")
    
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError(f"score_threshold must be between 0 and 1, got {score_threshold}")
    
    # Proceed with retrieval
    pass

# ❌ Bad: No validation, fails deep in code
def retrieve(query, top_k=5, score_threshold=0.7):
    # May crash with cryptic errors later
    pass
```

## Logging Standards

### Logger Setup

```python
import logging

# ✅ Good: Module-level logger
logger = logging.getLogger(__name__)

# Configure in main entry point only (main.py or app.py)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Log Levels

**Use appropriate log levels:**

```python
# DEBUG: Detailed diagnostic information
logger.debug(f"Retrieved {len(docs)} documents for query: {query}")
logger.debug(f"State after processing: {state}")

# INFO: Confirmation that things work as expected
logger.info(f"Starting sync for space: {space_key}")
logger.info(f"Processed page: {title} ({len(chunks)} chunks)")
logger.info(f"Request completed in {duration:.2f}s")

# WARNING: Something unexpected but application continues
logger.warning(f"Page {page_id} not found, skipping")
logger.warning(f"Low confidence ({confidence:.2f}), escalating")

# ERROR: Serious problem, application can recover
logger.error(f"Failed to process page {page_id}: {e}", exc_info=True)
logger.error(f"Vector store query failed: {e}", exc_info=True)

# CRITICAL: Very serious error, application may crash
logger.critical(f"Database connection failed: {e}", exc_info=True)
logger.critical(f"Cannot initialize required services")
```

### Structured Logging

```python
# ✅ Good: Include context
logger.info(f"Page processed: {title} (id={page_id}, chunks={len(chunks)})")
logger.error(f"Failed to fetch page {page_id} from Confluence: {e}", exc_info=True)

# ✅ Good: Use exc_info for exceptions
try:
    operation()
except Exception as e:
    logger.error(f"Operation failed: {e}", exc_info=True)  # Includes traceback

# ❌ Bad: Print statements
print(f"Processed {page_id}")  # Never use print()

# ❌ Bad: Missing context
logger.error("Failed to process page")  # Which page? What error?
```

### Never Log Sensitive Data

```python
# ✅ Good: Don't log secrets
logger.info(f"Connecting to Confluence at {base_url}")
logger.info(f"Using LLM model: {model_name}")

# ❌ Bad: Logging sensitive data
logger.info(f"Using token: {token}")  # DON'T LOG TOKENS
logger.debug(f"API key: {api_key}")  # DON'T LOG KEYS
logger.info(f"Password: {password}")  # DON'T LOG PASSWORDS
```

## Common Pitfalls to Avoid

### ❌ Don't Do This

```python
# 1. No print statements
print("Processing page")  # Use logger.info()

# 2. No bare except
try:
    risky_operation()
except:  # Too broad, hides errors
    pass

# 3. No mutable defaults
def process_pages(page_ids: list[str] = []):  # Bug! Mutable default
    page_ids.append("new")
    return page_ids

# 4. No direct env access
import os
token = os.getenv("CONFLUENCE_TOKEN")  # Use get_settings()

# 5. No missing type hints
def process(page_id):  # Add types!
    pass

# 6. No magic numbers
chunks = split_text(text, 1000)  # Use named constant

# 7. No catching Exception without re-raising
try:
    operation()
except Exception:
    pass  # Error lost

# 8. No missing docstrings on public functions
def public_function(arg):  # Add docstring
    pass

# 9. No mixing sync and async incorrectly
async def async_func():
    result = blocking_call()  # Blocks event loop!
```

### ✅ Do This Instead

```python
# 1. Use logging
logger.info("Processing page")

# 2. Catch specific exceptions
try:
    risky_operation()
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    raise
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    raise

# 3. Use None as default
def process_pages(page_ids: list[str] | None = None) -> list[str]:
    if page_ids is None:
        page_ids = []
    page_ids.append("new")
    return page_ids

# 4. Use settings
from hr_chatbot_config import get_settings
settings = get_settings()
token = settings.confluence.token

# 5. Add type hints
def process(page_id: str) -> dict[str, Any]:
    pass

# 6. Use named constants
CHUNK_SIZE = 1000
chunks = split_text(text, CHUNK_SIZE)

# 7. Always log and re-raise
try:
    operation()
except Exception as e:
    logger.exception("Operation failed")
    raise

# 8. Add docstrings
def public_function(arg: str) -> int:
    """Brief description.
    
    Args:
        arg: Description
        
    Returns:
        Description
    """
    pass

# 9. Use run_in_executor for blocking calls
import asyncio

async def async_func():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, blocking_call)
```

## Code Review Checklist

Before submitting code, verify:

- [ ] All functions have type hints
- [ ] All public functions have docstrings (Google style)
- [ ] No `print()` statements (use `logger`)
- [ ] No bare `except` clauses
- [ ] No direct `os.getenv()` calls (use `get_settings()`)
- [ ] Errors are logged with context before raising
- [ ] Constants are named (no magic numbers)
- [ ] Imports are organized (stdlib → third-party → local)
- [ ] Line length < 100 characters
- [ ] No mutable default arguments
- [ ] Environment-aware LLM/embedding selection (check `settings.llm.is_local`)
- [ ] Input validation at function entry
- [ ] No sensitive data (tokens, keys, passwords) in logs
- [ ] Specific exception types (not bare `Exception`)

## Domain-Specific Patterns

For patterns specific to:
- **LangGraph agents** → See `.claude/skills/langgraph-patterns/SKILL.md`
- **RAG pipeline** → See `.claude/skills/rag-pipeline/SKILL.md`
- **FastAPI endpoints** → See `.claude/skills/fastapi-patterns/SKILL.md`
- **Database operations** → See `.claude/skills/database-patterns/SKILL.md`