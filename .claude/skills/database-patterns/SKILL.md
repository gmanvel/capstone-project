# Database Patterns

## Overview

This skill provides patterns for database operations using SQLAlchemy with PostgreSQL.

**Scope**: General database patterns applicable across the entire project (API, Background Job, RAG Pipeline).

## Technology Recommendations

### ORM: SQLAlchemy 2.0+
- **Why**: Industry standard, type-safe, well-documented
- **Usage**: All database operations

### Migrations: SQLAlchemy `create_all()` (Recommended for this project)
- **Why**: Simple project with stable schema (1 table)
- **Alternative**: Alembic (if schema changes frequently or multiple developers)

### Connection Pooling: SQLAlchemy built-in
- **Default pool size**: 5 connections
- **Max overflow**: 10 connections
- **Good for**: This project's scale

### Query Builder: SQLAlchemy ORM (not raw SQL)
- **Why**: Type-safe, prevents SQL injection, easier to maintain
- **Exception**: Complex analytics queries can use raw SQL with parameterization

## SQLAlchemy Model Pattern

### Model Definition

```python
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, Index
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ConfluencePage(Base):
    """Confluence page tracking model.
    
    Tracks synchronized pages for idempotency and change detection.
    """
    
    __tablename__ = "confluence_pages"
    
    # Columns with documentation
    id = Column(String, primary_key=True, doc="Confluence page ID")
    title = Column(String, nullable=False, doc="Page title")
    space_key = Column(String, nullable=False, doc="Space key")
    content_hash = Column(String, nullable=False, doc="SHA256 hash of content")
    last_updated = Column(DateTime, nullable=False, doc="Last modified timestamp from Confluence")
    version = Column(Integer, nullable=False, doc="Page version number")
    url = Column(String, nullable=False, doc="Full page URL")
    synced_at = Column(DateTime, default=datetime.utcnow, doc="When we synced this page")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_space_key', 'space_key'),
        Index('idx_last_updated', 'last_updated'),
    )
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"<ConfluencePage(id={self.id}, title={self.title})>"
```

**Key Points:**
- Use `Base = declarative_base()` at module level
- Document columns with `doc` parameter
- Add indexes for frequently queried columns
- Implement `__repr__` for debugging
- Use appropriate column types (String, DateTime, Integer, Boolean)

### Column Types Reference

```python
from sqlalchemy import String, Integer, Float, Boolean, DateTime, Text, JSON

# Common types
id = Column(String, primary_key=True)           # Short strings (IDs, keys)
title = Column(String(255), nullable=False)     # Strings with max length
content = Column(Text)                          # Long text (no length limit)
count = Column(Integer, default=0)              # Integers
score = Column(Float)                           # Decimals
is_active = Column(Boolean, default=True)       # Booleans
created_at = Column(DateTime, default=datetime.utcnow)  # Timestamps
metadata = Column(JSON)                         # JSON data (Postgres JSONB)
```

## Database Initialization Pattern

### Using `create_all()` (Recommended)

```python
from sqlalchemy import create_engine
from hr_chatbot_config import get_settings
from .models import Base
import logging

logger = logging.getLogger(__name__)

def init_database() -> None:
    """Initialize database and create tables.
    
    Creates all tables defined in SQLAlchemy models.
    Idempotent: safe to run multiple times.
    """
    settings = get_settings()
    
    logger.info("Initializing database")
    engine = create_engine(settings.database.connection_string)
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    logger.info("Database initialized successfully")


# Usage in main.py or app startup
if __name__ == "__main__":
    init_database()
    # Continue with application
```

**When to call:**
- On first deployment
- In Docker entrypoint script
- At application startup (safe - it's idempotent)

### Using Alembic (Alternative for production)

**Setup:**
```bash
# Install
pip install alembic

# Initialize
alembic init alembic

# Create migration
alembic revision --autogenerate -m "create confluence_pages table"

# Apply migration
alembic upgrade head
```

**When to use Alembic:**
- Multiple developers working on schema
- Need rollback capability
- Frequent schema changes
- Production deployments requiring audit trail

**For this project:** Not necessary, but good to know.

## Session Management Patterns

### Context Manager Pattern (Recommended)

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from hr_chatbot_config import get_settings
import logging

logger = logging.getLogger(__name__)

# Module-level engine (create once)
_engine = None

def get_engine():
    """Get or create database engine singleton."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.database.connection_string,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True  # Verify connections before using
        )
    return _engine


@contextmanager
def get_db_session() -> Session:
    """Context manager for database sessions.
    
    Automatically commits on success, rolls back on error.
    Always closes session.
    
    Usage:
        with get_db_session() as session:
            page = session.query(ConfluencePage).first()
            # Auto-commit on success
    
    Yields:
        SQLAlchemy session
    """
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}", exc_info=True)
        raise
    finally:
        session.close()


# Usage example
with get_db_session() as session:
    page = ConfluencePage(
        id="123",
        title="Test",
        space_key="HR",
        content_hash="abc",
        last_updated=datetime.utcnow(),
        version=1,
        url="https://..."
    )
    session.add(page)
    # Auto-commit on exit
```

### Dependency Injection Pattern (FastAPI)

```python
from fastapi import Depends
from sqlalchemy.orm import Session

def get_db() -> Session:
    """Dependency for FastAPI endpoints.
    
    Yields database session that auto-closes after request.
    """
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Usage in FastAPI endpoint
from fastapi import APIRouter, Depends

router = APIRouter()

@router.get("/pages")
def list_pages(db: Session = Depends(get_db)):
    """List all pages."""
    pages = db.query(ConfluencePage).all()
    return {"pages": [p.title for p in pages]}
```

**Key Points:**
- Use context manager for scripts and background jobs
- Use dependency injection for FastAPI endpoints
- Never commit/rollback in endpoint code - let FastAPI handle it
- Always close sessions

## Query Patterns

### Basic Queries

```python
from sqlalchemy.orm import Session
from .models import ConfluencePage

# Get by primary key
page = session.query(ConfluencePage).filter_by(id="123").first()

# Get by primary key (alternative)
page = session.get(ConfluencePage, "123")

# Get all
pages = session.query(ConfluencePage).all()

# Get with filter
hr_pages = session.query(ConfluencePage).filter_by(space_key="HR").all()

# Get with complex filter
recent_pages = session.query(ConfluencePage).filter(
    ConfluencePage.space_key == "HR",
    ConfluencePage.last_updated > datetime(2025, 1, 1)
).all()

# Count
count = session.query(ConfluencePage).count()

# Exists check
exists = session.query(ConfluencePage).filter_by(id="123").first() is not None

# Order by
pages = session.query(ConfluencePage).order_by(
    ConfluencePage.last_updated.desc()
).all()

# Limit
recent = session.query(ConfluencePage).order_by(
    ConfluencePage.last_updated.desc()
).limit(10).all()
```

### Safe Query Pattern with Error Handling

```python
def get_page(session: Session, page_id: str) -> ConfluencePage | None:
    """Get page by ID.
    
    Args:
        session: Database session
        page_id: Confluence page ID
        
    Returns:
        Page if found, None otherwise
        
    Raises:
        SQLAlchemyError: If database query fails
    """
    try:
        page = session.query(ConfluencePage).filter_by(id=page_id).first()
        return page
    except SQLAlchemyError as e:
        logger.error(f"Failed to query page {page_id}: {e}", exc_info=True)
        raise


def get_pages_by_space(session: Session, space_key: str) -> list[ConfluencePage]:
    """Get all pages in a space.
    
    Args:
        session: Database session
        space_key: Confluence space key
        
    Returns:
        List of pages (empty if none found)
    """
    try:
        pages = session.query(ConfluencePage).filter_by(space_key=space_key).all()
        logger.info(f"Found {len(pages)} pages in space {space_key}")
        return pages
    except SQLAlchemyError as e:
        logger.error(f"Failed to query pages in space {space_key}: {e}", exc_info=True)
        raise
```

### Advanced Query Patterns

```python
# Multiple filters with AND
pages = session.query(ConfluencePage).filter(
    ConfluencePage.space_key == "HR",
    ConfluencePage.version > 5
).all()

# OR conditions
from sqlalchemy import or_

pages = session.query(ConfluencePage).filter(
    or_(
        ConfluencePage.space_key == "HR",
        ConfluencePage.space_key == "ENG"
    )
).all()

# IN clause
pages = session.query(ConfluencePage).filter(
    ConfluencePage.space_key.in_(["HR", "ENG", "FIN"])
).all()

# LIKE pattern matching
pages = session.query(ConfluencePage).filter(
    ConfluencePage.title.like("%policy%")
).all()

# NULL checks
pages = session.query(ConfluencePage).filter(
    ConfluencePage.synced_at.is_(None)
).all()

# Date range
from datetime import datetime, timedelta

start_date = datetime.utcnow() - timedelta(days=7)
recent_pages = session.query(ConfluencePage).filter(
    ConfluencePage.last_updated >= start_date
).all()

# Distinct
space_keys = session.query(ConfluencePage.space_key).distinct().all()

# Aggregate functions
from sqlalchemy import func

count_by_space = session.query(
    ConfluencePage.space_key,
    func.count(ConfluencePage.id)
).group_by(ConfluencePage.space_key).all()
```

## Insert/Update/Delete Patterns

### Insert (Create)

```python
def create_page(session: Session, page_data: dict) -> ConfluencePage:
    """Create new page record.
    
    Args:
        session: Database session
        page_data: Page data dictionary
        
    Returns:
        Created page object
    """
    page = ConfluencePage(
        id=page_data["id"],
        title=page_data["title"],
        space_key=page_data["space_key"],
        content_hash=page_data["content_hash"],
        last_updated=page_data["last_updated"],
        version=page_data["version"],
        url=page_data["url"]
    )
    
    session.add(page)
    # Don't commit here - let context manager handle it
    
    logger.info(f"Created page {page.id}")
    return page
```

### Update

```python
def update_page(session: Session, page_id: str, updates: dict) -> ConfluencePage:
    """Update existing page.
    
    Args:
        session: Database session
        page_id: Page ID to update
        updates: Dictionary of fields to update
        
    Returns:
        Updated page object
        
    Raises:
        ValueError: If page not found
    """
    page = session.query(ConfluencePage).filter_by(id=page_id).first()
    
    if not page:
        raise ValueError(f"Page {page_id} not found")
    
    # Update fields
    for key, value in updates.items():
        if hasattr(page, key):
            setattr(page, key, value)
    
    page.synced_at = datetime.utcnow()
    
    logger.info(f"Updated page {page_id}")
    return page
```

### Upsert (Insert or Update)

```python
def upsert_page(session: Session, page_data: dict) -> ConfluencePage:
    """Insert or update page.
    
    Args:
        session: Database session
        page_data: Page data dictionary
        
    Returns:
        Page object (created or updated)
    """
    existing = session.query(ConfluencePage).filter_by(id=page_data["id"]).first()
    
    if existing:
        # Update
        logger.info(f"Updating existing page {page_data['id']}")
        for key, value in page_data.items():
            if hasattr(existing, key):
                setattr(existing, key, value)
        existing.synced_at = datetime.utcnow()
        return existing
    else:
        # Insert
        logger.info(f"Creating new page {page_data['id']}")
        page = ConfluencePage(**page_data)
        session.add(page)
        return page
```

### Delete

```python
def delete_page(session: Session, page_id: str) -> bool:
    """Delete page by ID.
    
    Args:
        session: Database session
        page_id: Page ID to delete
        
    Returns:
        True if deleted, False if not found
    """
    page = session.query(ConfluencePage).filter_by(id=page_id).first()
    
    if not page:
        logger.warning(f"Page {page_id} not found for deletion")
        return False
    
    session.delete(page)
    logger.info(f"Deleted page {page_id}")
    return True


def delete_pages_by_space(session: Session, space_key: str) -> int:
    """Delete all pages in a space.
    
    Args:
        session: Database session
        space_key: Space key
        
    Returns:
        Number of pages deleted
    """
    count = session.query(ConfluencePage).filter_by(space_key=space_key).delete()
    logger.info(f"Deleted {count} pages from space {space_key}")
    return count
```

## Transaction Patterns

### Explicit Transaction

```python
def batch_update(session: Session, page_ids: list[str], updates: dict) -> None:
    """Update multiple pages in a transaction.
    
    All updates succeed or all fail (atomic).
    
    Args:
        session: Database session
        page_ids: List of page IDs to update
        updates: Fields to update
        
    Raises:
        Exception: If any update fails, all are rolled back
    """
    try:
        for page_id in page_ids:
            page = session.query(ConfluencePage).filter_by(id=page_id).first()
            if page:
                for key, value in updates.items():
                    setattr(page, key, value)
        
        # Explicit commit (if not using context manager)
        # session.commit()
        
        logger.info(f"Updated {len(page_ids)} pages")
    
    except Exception as e:
        logger.error(f"Batch update failed: {e}", exc_info=True)
        # Explicit rollback (if not using context manager)
        # session.rollback()
        raise
```

### Savepoints (Nested Transactions)

```python
def process_with_savepoint(session: Session, page_id: str) -> None:
    """Process page with savepoint for partial rollback.
    
    If processing fails, only this operation rolls back,
    not the entire transaction.
    """
    # Create savepoint
    savepoint = session.begin_nested()
    
    try:
        # Risky operation
        page = session.query(ConfluencePage).filter_by(id=page_id).first()
        page.version += 1
        
        savepoint.commit()
        logger.info(f"Processed page {page_id}")
    
    except Exception as e:
        savepoint.rollback()
        logger.warning(f"Failed to process {page_id}, rolled back: {e}")
        # Main transaction continues
```

## Connection Pool Configuration

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

def create_engine_with_pool():
    """Create engine with custom connection pool."""
    settings = get_settings()
    
    engine = create_engine(
        settings.database.connection_string,
        
        # Pool settings
        poolclass=QueuePool,
        pool_size=5,              # Number of connections to maintain
        max_overflow=10,          # Max additional connections when pool full
        pool_timeout=30,          # Seconds to wait for connection
        pool_recycle=3600,        # Recycle connections after 1 hour
        pool_pre_ping=True,       # Test connections before using
        
        # Connection settings
        connect_args={
            "connect_timeout": 10,
            "application_name": "hr_chatbot"
        },
        
        # Echo SQL for debugging (disable in production)
        echo=False
    )
    
    return engine
```

## Best Practices

### ✅ Do This

```python
# 1. Use context manager for sessions
with get_db_session() as session:
    page = session.query(ConfluencePage).first()

# 2. Use ORM queries (not raw SQL)
pages = session.query(ConfluencePage).filter_by(space_key="HR").all()

# 3. Use filter_by for simple equality
page = session.query(ConfluencePage).filter_by(id=page_id).first()

# 4. Handle None returns
page = session.query(ConfluencePage).filter_by(id="unknown").first()
if page is None:
    logger.warning("Page not found")

# 5. Log database operations
logger.info(f"Querying pages in space {space_key}")

# 6. Use pool_pre_ping for reliability
engine = create_engine(url, pool_pre_ping=True)

# 7. Validate before database operations
if not page_id:
    raise ValueError("page_id is required")
```

### ❌ Don't Do This

```python
# 1. Don't use raw SQL (SQL injection risk)
session.execute(f"SELECT * FROM pages WHERE id = '{page_id}'")  # UNSAFE

# 2. Don't forget to close sessions
session = Session()
# ... do work
# session not closed - memory leak

# 3. Don't catch exceptions without logging
try:
    session.query(ConfluencePage).all()
except:
    pass  # Silent failure

# 4. Don't commit in every function
def get_page(session):
    page = session.query(ConfluencePage).first()
    session.commit()  # Unnecessary commit
    return page

# 5. Don't create engine in every function
def get_data():
    engine = create_engine(url)  # Recreating engine
    # ...

# 6. Don't use string concatenation in queries
session.query(ConfluencePage).filter(f"id = '{page_id}'")  # UNSAFE
```

## Raw SQL (When Necessary)

**Use only for complex queries that are hard to express in ORM.**

```python
from sqlalchemy import text

def complex_analytics(session: Session) -> list[dict]:
    """Complex query requiring raw SQL.
    
    Always use parameterized queries to prevent SQL injection.
    """
    query = text("""
        SELECT 
            space_key,
            COUNT(*) as page_count,
            AVG(version) as avg_version
        FROM confluence_pages
        WHERE last_updated > :start_date
        GROUP BY space_key
        HAVING COUNT(*) > :min_count
    """)
    
    result = session.execute(
        query,
        {
            "start_date": datetime(2025, 1, 1),
            "min_count": 5
        }
    )
    
    return [dict(row) for row in result]
```

**Key**: Always use `:parameter` syntax and pass params dict - never string concatenation.

## Common Patterns Summary

| Pattern | Use Case | Implementation |
|---------|----------|----------------|
| Context Manager | Scripts, background jobs | `with get_db_session() as session:` |
| Dependency Injection | FastAPI endpoints | `Depends(get_db)` |
| Upsert | Insert or update | Check exists, then add or update |
| Bulk Insert | Many records at once | `session.bulk_insert_mappings()` |
| Pagination | Large result sets | `.limit(n).offset(m)` |
| Soft Delete | Mark deleted, don't remove | Add `is_deleted` column |

## Testing Database Code

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture
def test_db():
    """Create in-memory test database."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    yield session
    
    session.close()


def test_create_page(test_db):
    """Test page creation."""
    page = ConfluencePage(
        id="test",
        title="Test Page",
        space_key="TEST",
        content_hash="abc",
        last_updated=datetime.utcnow(),
        version=1,
        url="https://test"
    )
    
    test_db.add(page)
    test_db.commit()
    
    result = test_db.query(ConfluencePage).filter_by(id="test").first()
    assert result is not None
    assert result.title == "Test Page"
```