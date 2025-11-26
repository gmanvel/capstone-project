"""Tests for database models using testcontainers.

This module tests the ConfluencePage model and database initialization
using a real PostgreSQL container via testcontainers. This ensures that
the schema, indexes, and operations work correctly in a real database.
"""

import os
from datetime import datetime, timezone
from hashlib import sha256

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker
from testcontainers.postgres import PostgresContainer

from rag_pipeline.models import Base, ConfluencePage, init_database


@pytest.fixture(scope="module")
def postgres_container():
    """Start a PostgreSQL container for testing.
    
    Yields:
        PostgresContainer: Running PostgreSQL container
    """
    with PostgresContainer("postgres:16-alpine") as postgres:
        yield postgres


@pytest.fixture(scope="module")
def database_url(postgres_container):
    """Get database connection URL from container.
    
    Args:
        postgres_container: PostgreSQL container fixture
        
    Returns:
        str: SQLAlchemy connection string
    """
    return postgres_container.get_connection_url()


@pytest.fixture(scope="module")
def engine(database_url):
    """Create SQLAlchemy engine for the test database.
    
    Args:
        database_url: Database connection string
        
    Returns:
        Engine: SQLAlchemy engine instance
    """
    return create_engine(database_url)


@pytest.fixture(scope="module", autouse=True)
def setup_database(database_url, engine):
    """Initialize database schema before tests.
    
    This fixture sets the database URL in environment variables
    so that init_database() can pick it up via settings.
    
    Args:
        database_url: Database connection string
        engine: SQLAlchemy engine
    """
    # Parse connection URL to set individual env vars for settings
    # Format: postgresql://user:password@host:port/database
    from urllib.parse import urlparse
    
    parsed = urlparse(database_url)
    os.environ["POSTGRES_USER"] = parsed.username
    os.environ["POSTGRES_PASSWORD"] = parsed.password
    os.environ["POSTGRES_HOST"] = parsed.hostname
    os.environ["POSTGRES_PORT"] = str(parsed.port)
    os.environ["POSTGRES_DB"] = parsed.path.lstrip("/")
    
    # Initialize the database
    init_database()
    
    yield
    
    # Cleanup
    Base.metadata.drop_all(engine)


@pytest.fixture
def db_session(engine):
    """Create a new database session for each test.
    
    Args:
        engine: SQLAlchemy engine
        
    Yields:
        Session: Database session that's rolled back after test
    """
    connection = engine.connect()
    transaction = connection.begin()
    SessionLocal = sessionmaker(bind=connection)
    session = SessionLocal()
    
    yield session
    
    session.close()
    # Only rollback if transaction is still active
    if transaction.is_active:
        transaction.rollback()
    connection.close()


class TestDatabaseInitialization:
    """Tests for database initialization and schema."""
    
    def test_tables_created(self, engine):
        """Verify that confluence_pages table is created."""
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        assert "confluence_pages" in tables
    
    def test_table_columns(self, engine):
        """Verify all required columns exist with correct types."""
        inspector = inspect(engine)
        columns = {col["name"]: col for col in inspector.get_columns("confluence_pages")}
        
        # Check all required columns exist
        required_columns = {
            "id", "title", "space_key", "content_hash",
            "last_updated", "version", "url", "synced_at"
        }
        assert set(columns.keys()) == required_columns
        
        # Check column types
        assert columns["id"]["type"].__class__.__name__ in ["VARCHAR", "TEXT", "String"]
        assert columns["version"]["type"].__class__.__name__ == "INTEGER"
        assert columns["last_updated"]["type"].__class__.__name__ == "TIMESTAMP"
        assert columns["last_updated"]["type"].timezone is True
        assert columns["synced_at"]["type"].__class__.__name__ == "TIMESTAMP"
        assert columns["synced_at"]["type"].timezone is True
    
    def test_primary_key(self, engine):
        """Verify primary key is set on id column."""
        inspector = inspect(engine)
        pk = inspector.get_pk_constraint("confluence_pages")
        
        assert pk["constrained_columns"] == ["id"]
    
    def test_indexes_created(self, engine):
        """Verify indexes are created on space_key and last_updated."""
        inspector = inspect(engine)
        indexes = {idx["name"]: idx for idx in inspector.get_indexes("confluence_pages")}
        
        # Check that our indexes exist
        assert "idx_space_key" in indexes
        assert "idx_last_updated" in indexes
        
        # Verify index columns
        assert indexes["idx_space_key"]["column_names"] == ["space_key"]
        assert indexes["idx_last_updated"]["column_names"] == ["last_updated"]


class TestConfluencePageModel:
    """Tests for ConfluencePage model CRUD operations."""
    
    def test_create_confluence_page(self, db_session: Session):
        """Test creating a new Confluence page record."""
        page = ConfluencePage(
            id="12345",
            title="Test Page",
            space_key="HR",
            content_hash=sha256(b"test content").hexdigest(),
            last_updated=datetime(2025, 11, 17, 10, 0, 0, tzinfo=timezone.utc),
            version=1,
            url="https://confluence.example.com/pages/12345"
        )
        
        db_session.add(page)
        db_session.commit()
        
        # Verify it was saved
        retrieved = db_session.query(ConfluencePage).filter_by(id="12345").first()
        assert retrieved is not None
        assert retrieved.title == "Test Page"
        assert retrieved.space_key == "HR"
        assert retrieved.version == 1
    
    def test_update_confluence_page(self, db_session: Session):
        """Test updating an existing Confluence page."""
        # Create initial page
        page = ConfluencePage(
            id="67890",
            title="Original Title",
            space_key="HR",
            content_hash=sha256(b"original").hexdigest(),
            last_updated=datetime(2025, 11, 17, 10, 0, 0, tzinfo=timezone.utc),
            version=1,
            url="https://confluence.example.com/pages/67890"
        )
        db_session.add(page)
        db_session.commit()
        
        # Update it
        page.title = "Updated Title"
        page.version = 2
        page.content_hash = sha256(b"updated content").hexdigest()
        db_session.commit()
        
        # Verify update
        retrieved = db_session.query(ConfluencePage).filter_by(id="67890").first()
        assert retrieved.title == "Updated Title"
        assert retrieved.version == 2
    
    def test_query_by_space_key(self, db_session: Session):
        """Test querying pages by space_key (indexed column)."""
        # Create pages in different spaces
        hr_page = ConfluencePage(
            id="hr1",
            title="HR Policy",
            space_key="HR",
            content_hash=sha256(b"hr content").hexdigest(),
            last_updated=datetime(2025, 11, 17, 10, 0, 0, tzinfo=timezone.utc),
            version=1,
            url="https://confluence.example.com/pages/hr1"
        )
        eng_page = ConfluencePage(
            id="eng1",
            title="Engineering Doc",
            space_key="ENG",
            content_hash=sha256(b"eng content").hexdigest(),
            last_updated=datetime(2025, 11, 17, 10, 0, 0, tzinfo=timezone.utc),
            version=1,
            url="https://confluence.example.com/pages/eng1"
        )
        
        db_session.add_all([hr_page, eng_page])
        db_session.commit()
        
        # Query by space_key
        hr_pages = db_session.query(ConfluencePage).filter_by(space_key="HR").all()
        assert len(hr_pages) == 1
        assert hr_pages[0].id == "hr1"
    
    def test_duplicate_id_fails(self, db_session: Session):
        """Test that duplicate page IDs are rejected (primary key constraint)."""
        page1 = ConfluencePage(
            id="dup123",
            title="First Page",
            space_key="HR",
            content_hash=sha256(b"content1").hexdigest(),
            last_updated=datetime(2025, 11, 17, 10, 0, 0, tzinfo=timezone.utc),
            version=1,
            url="https://confluence.example.com/pages/dup123"
        )
        db_session.add(page1)
        db_session.commit()
        
        # Expunge the first page from the session to avoid identity conflicts
        db_session.expunge(page1)
        
        # Try to add another page with same ID
        page2 = ConfluencePage(
            id="dup123",
            title="Second Page",
            space_key="HR",
            content_hash=sha256(b"content2").hexdigest(),
            last_updated=datetime(2025, 11, 17, 10, 0, 0, tzinfo=timezone.utc),
            version=1,
            url="https://confluence.example.com/pages/dup123-2"
        )
        db_session.add(page2)
        
        with pytest.raises(Exception):  # Will raise IntegrityError
            db_session.commit()
        
        # Explicitly rollback to clean up session state before fixture cleanup
        db_session.rollback()
    
    def test_synced_at_defaults(self, db_session: Session):
        """Test that synced_at has a default value."""
        page = ConfluencePage(
            id="auto123",
            title="Auto Timestamp",
            space_key="HR",
            content_hash=sha256(b"content").hexdigest(),
            last_updated=datetime(2025, 11, 17, 10, 0, 0, tzinfo=timezone.utc),
            version=1,
            url="https://confluence.example.com/pages/auto123"
        )
        
        db_session.add(page)
        db_session.commit()
        
        # Verify synced_at was set automatically
        retrieved = db_session.query(ConfluencePage).filter_by(id="auto123").first()
        assert retrieved.synced_at is not None
        assert isinstance(retrieved.synced_at, datetime)
    
    def test_repr(self, db_session: Session):
        """Test string representation of ConfluencePage."""
        page = ConfluencePage(
            id="repr123",
            title="Repr Test",
            space_key="HR",
            content_hash=sha256(b"content").hexdigest(),
            last_updated=datetime(2025, 11, 17, 10, 0, 0, tzinfo=timezone.utc),
            version=1,
            url="https://confluence.example.com/pages/repr123"
        )
        
        repr_str = repr(page)
        assert "repr123" in repr_str
        assert "Repr Test" in repr_str
        assert "ConfluencePage" in repr_str


class TestIdempotentSync:
    """Tests for idempotent sync operations using content hashes."""
    
    def test_content_hash_change_detection(self, db_session: Session):
        """Test that content hash changes can be detected for re-indexing."""
        original_content = b"Original content"
        original_hash = sha256(original_content).hexdigest()
        
        # Create initial page
        page = ConfluencePage(
            id="hash123",
            title="Hash Test",
            space_key="HR",
            content_hash=original_hash,
            last_updated=datetime(2025, 11, 17, 10, 0, 0, tzinfo=timezone.utc),
            version=1,
            url="https://confluence.example.com/pages/hash123"
        )
        db_session.add(page)
        db_session.commit()
        
        # Simulate checking if page needs update
        retrieved = db_session.query(ConfluencePage).filter_by(id="hash123").first()
        
        # New content with different hash
        new_content = b"Updated content"
        new_hash = sha256(new_content).hexdigest()
        
        # Verify hashes are different (content changed)
        assert retrieved.content_hash != new_hash
        
        # Update with new hash (would trigger re-indexing in real sync)
        retrieved.content_hash = new_hash
        retrieved.version = 2
        db_session.commit()
        
        # Verify update
        updated = db_session.query(ConfluencePage).filter_by(id="hash123").first()
        assert updated.content_hash == new_hash
        assert updated.version == 2
