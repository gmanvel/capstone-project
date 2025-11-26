"""Configuration management for HR Chatbot.

This module provides centralized configuration using Pydantic Settings.
All services should use get_settings() instead of os.getenv() directly.
Configuration is loaded from environment variables and .env files.

Example:
    >>> from hr_chatbot_config import get_settings
    >>> settings = get_settings()
    >>> db_url = settings.database.connection_string
    >>> model_name = settings.llm.chat_model_name
"""

from enum import Enum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment enum."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"


class DatabaseConfig(BaseSettings):
    """PostgreSQL database configuration.

    Loads configuration from environment variables with POSTGRES_ prefix.
    Provides a computed connection string for SQLAlchemy.

    Attributes:
        host: Database host address
        port: Database port number
        db: Database name
        user: Database username
        password: Database password (required)
    """

    model_config = SettingsConfigDict(env_prefix="POSTGRES_")

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    db: str = Field(default="hrbot", description="Database name")
    user: str = Field(default="hrbot", description="Database username")
    password: str = Field(description="Database password (required)")

    @property
    def connection_string(self) -> str:
        """Build PostgreSQL connection URL.

        Returns:
            PostgreSQL connection string for SQLAlchemy
        """
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


class QdrantConfig(BaseSettings):
    """Qdrant vector database configuration.

    Loads configuration from environment variables with QDRANT_ prefix.
    Provides a computed URL for Qdrant client connection.

    Attributes:
        host: Qdrant host address
        port: Qdrant port number
        collection_name: Name of the vector collection
    """

    model_config = SettingsConfigDict(env_prefix="QDRANT_")

    host: str = Field(default="localhost", description="Qdrant host")
    port: int = Field(default=6333, description="Qdrant port")
    collection_name: str = Field(
        default="hr_documents",
        description="Vector collection name"
    )

    @property
    def url(self) -> str:
        """Build Qdrant connection URL.

        Returns:
            Qdrant HTTP API URL
        """
        return f"http://{self.host}:{self.port}"


class ConfluenceConfig(BaseSettings):
    """Confluence API configuration.

    Loads configuration from environment variables with CONFLUENCE_ prefix.

    Attributes:
        token: Confluence API token (required)
        base_url: Confluence instance base URL (required)
        space_key: Confluence space key to sync
    """

    model_config = SettingsConfigDict(env_prefix="CONFLUENCE_")

    token: str = Field(description="Confluence API token (required)")
    base_url: str = Field(description="Confluence base URL (required)")
    space_key: str = Field(default="HRS", description="Confluence space key")


class ChunkingConfig(BaseSettings):
    """Text chunking configuration.

    Loads configuration from environment variables with CHUNKING_ prefix.
    Controls semantic chunking behavior for document processing.

    Attributes:
        semantic_breakpoint_threshold: Percentile threshold for semantic chunk
            boundaries (50-100). Higher values create larger, more coherent chunks.
    """

    model_config = SettingsConfigDict(env_prefix="CHUNKING_")

    semantic_breakpoint_threshold: int = Field(
        default=95,
        ge=50,
        le=100,
        description="Percentile threshold for semantic chunk boundaries"
    )


class LLMConfig(BaseSettings):
    """LLM configuration with environment-aware model selection.

    Automatically switches between Ollama (development) and OpenAI (production)
    based on the environment setting. Provides computed properties for
    selecting the appropriate model names.

    Attributes:
        environment: Current application environment
        ollama_base_url: Ollama API base URL
        ollama_model: Ollama chat model name
        ollama_embedding_model: Ollama embedding model name
        openai_api_key: OpenAI API key (optional, required for production)
        openai_model: OpenAI chat model name
        openai_embedding_model: OpenAI embedding model name
    """

    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment"
    )

    # Ollama configuration (development)
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL"
    )
    ollama_model: str = Field(
        default="mistral",
        description="Ollama chat model"
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Ollama embedding model"
    )

    # OpenAI configuration (production)
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key"
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI chat model"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model"
    )

    @property
    def is_local(self) -> bool:
        """Check if using local Ollama models.

        Returns:
            True if environment is development, False otherwise
        """
        return self.environment == Environment.DEVELOPMENT

    @property
    def chat_model_name(self) -> str:
        """Get the appropriate chat model name for current environment.

        Returns:
            Ollama model name if development, OpenAI model name if production
        """
        return self.ollama_model if self.is_local else self.openai_model

    @property
    def embedding_model_name(self) -> str:
        """Get the appropriate embedding model name for current environment.

        Returns:
            Ollama embedding model if development, OpenAI embedding model
            if production
        """
        return (
            self.ollama_embedding_model
            if self.is_local
            else self.openai_embedding_model
        )


class MemoryConfig(BaseSettings):
    """Configuration for conversation memory feature.

    Controls behavior of session-based conversation memory including
    when to trigger summarization and memory size limits.

    Attributes:
        enabled: Enable/disable conversation memory feature
        message_threshold: Number of messages before triggering summarization
        max_summary_length: Maximum token length for conversation summaries
        summarization_temperature: LLM temperature for generating summaries
    """

    model_config = SettingsConfigDict(env_prefix="MEMORY_")

    enabled: bool = Field(
        default=True,
        description="Enable conversation memory feature"
    )

    message_threshold: int = Field(
        default=6,
        ge=2,
        le=20,
        description="Number of messages before triggering summarization"
    )

    max_summary_length: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Maximum token length for conversation summaries"
    )

    summarization_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="LLM temperature for generating summaries (lower = more focused)"
    )


class Settings(BaseSettings):
    """Master configuration class for HR Chatbot.

    Aggregates all configuration sections and provides access to them
    through properties. Loads configuration from environment variables
    and .env file.

    Attributes:
        environment: Current application environment (DEVELOPMENT/PRODUCTION)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"
    )

    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment"
    )

    @property
    def database(self) -> DatabaseConfig:
        """Get database configuration.

        Returns:
            DatabaseConfig instance with PostgreSQL settings
        """
        return DatabaseConfig()

    @property
    def qdrant(self) -> QdrantConfig:
        """Get Qdrant configuration.

        Returns:
            QdrantConfig instance with vector database settings
        """
        return QdrantConfig()

    @property
    def llm(self) -> LLMConfig:
        """Get LLM configuration.

        Returns:
            LLMConfig instance with environment-aware model settings
        """
        return LLMConfig(environment=self.environment)

    @property
    def confluence(self) -> ConfluenceConfig:
        """Get Confluence configuration.

        Returns:
            ConfluenceConfig instance with API credentials
        """
        return ConfluenceConfig()

    @property
    def chunking(self) -> ChunkingConfig:
        """Get chunking configuration.

        Returns:
            ChunkingConfig instance with text chunking settings
        """
        return ChunkingConfig()

    @property
    def memory(self) -> MemoryConfig:
        """Get memory configuration.

        Returns:
            MemoryConfig instance with conversation memory settings
        """
        return MemoryConfig()


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns a singleton Settings instance that is cached after first call.
    This ensures configuration is loaded only once and reused across the
    application.

    Returns:
        Settings instance with all configuration sections

    Example:
        >>> settings = get_settings()
        >>> db_url = settings.database.connection_string
        >>> is_dev = settings.llm.is_local
    """
    return Settings()


__all__ = [
    "Environment",
    "DatabaseConfig",
    "QdrantConfig",
    "ConfluenceConfig",
    "ChunkingConfig",
    "LLMConfig",
    "MemoryConfig",
    "Settings",
    "get_settings",
]
