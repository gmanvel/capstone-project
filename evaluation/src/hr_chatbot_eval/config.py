"""RAGAS configuration with Ollama support."""

from hr_chatbot_config import get_settings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


def get_ragas_llm() -> LangchainLLMWrapper:
    """Get RAGAS-wrapped LLM based on environment settings.

    Returns:
        LangchainLLMWrapper: RAGAS-compatible LLM wrapper
    """
    settings = get_settings()

    if settings.llm.is_local:
        llm = ChatOllama(
            base_url=settings.llm.ollama_base_url,
            model=settings.llm.chat_model_name,
            temperature=0.0,  # Deterministic for evaluation
        )
    else:
        llm = ChatOpenAI(
            api_key=settings.llm.openai_api_key,
            model=settings.llm.chat_model_name,
            temperature=0.0,
        )

    return LangchainLLMWrapper(llm)


def get_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """Get RAGAS-wrapped embeddings based on environment settings.

    Returns:
        LangchainEmbeddingsWrapper: RAGAS-compatible embeddings wrapper
    """
    settings = get_settings()

    if settings.llm.is_local:
        embeddings = OllamaEmbeddings(
            base_url=settings.llm.ollama_base_url,
            model=settings.llm.embedding_model_name,
        )
    else:
        embeddings = OpenAIEmbeddings(
            api_key=settings.llm.openai_api_key,
            model=settings.llm.embedding_model_name,
        )

    return LangchainEmbeddingsWrapper(embeddings)
