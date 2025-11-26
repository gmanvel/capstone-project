"""RAG Pipeline package for HR Chatbot.

This package provides database models and utilities for the RAG pipeline,
including Confluence page tracking and vector storage integration.
"""

from rag_pipeline.chunker_service import ChunkerService
from rag_pipeline.confluence_client import ConfluenceClient
from rag_pipeline.models import Base, ConfluencePage, init_database
from rag_pipeline.rag_pipeline import RAGPipeline, SyncResult

__all__ = [
    "Base",
    "ChunkerService",
    "ConfluenceClient",
    "ConfluencePage",
    "RAGPipeline",
    "SyncResult",
    "init_database",
]
