"""Test configuration module."""

import os
import pytest
from hr_chatbot_config import get_settings, Environment


@pytest.fixture(autouse=True)
def set_test_env():
    """Set minimal required environment variables for testing."""
    os.environ["POSTGRES_PASSWORD"] = "test_pass"
    yield


def test_importable():
    """Verify module is importable."""
    from hr_chatbot_config import Settings, DatabaseConfig, QdrantConfig, LLMConfig, ConfluenceConfig
    assert Settings is not None


def test_settings_instance():
    """Verify settings instance creation."""
    settings = get_settings()
    assert settings is not None
    assert settings.environment == Environment.DEVELOPMENT


def test_database_config():
    """Verify database configuration bindings."""
    settings = get_settings()
    db = settings.database
    assert db.host == "localhost"
    assert db.port == 5432
    assert db.db == "hrbot"
    assert db.user == "hrbot"
    assert db.password == "test_pass"
    assert "postgresql://" in db.connection_string


def test_qdrant_config():
    """Verify Qdrant configuration bindings."""
    settings = get_settings()
    qdrant = settings.qdrant
    assert qdrant.host == "localhost"
    assert qdrant.port == 6333
    assert qdrant.collection_name == "hr_documents"
    assert qdrant.url == "http://localhost:6333"


def test_llm_config():
    """Verify LLM configuration bindings."""
    settings = get_settings()
    llm = settings.llm
    assert llm.environment == Environment.DEVELOPMENT
    assert llm.is_local is True
    assert llm.chat_model_name == "mistral"
    assert llm.embedding_model_name == "nomic-embed-text"


def test_singleton_cache():
    """Verify settings singleton behavior."""
    settings1 = get_settings()
    settings2 = get_settings()
    assert settings1 is settings2
