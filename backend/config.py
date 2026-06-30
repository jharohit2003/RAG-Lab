"""config.py -- swappable embedding + chat providers (lazy imports)."""

from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local").lower()
    local_embedding_model: str = os.getenv(
        "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    openai_embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    voyage_embedding_model: str = os.getenv("VOYAGE_EMBEDDING_MODEL", "voyage-3-lite")

    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic").lower()
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_embedding_model: str = os.getenv(
        "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
    )


settings = Settings()


def get_embeddings():
    """Return (embeddings_object, label). Embeddings are required, so raise if a key is missing."""
    provider = settings.embedding_provider
    if provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

        m = HuggingFaceEmbeddings(
            model_name=settings.local_embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        return m, f"local:{settings.local_embedding_model}"
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        m = OllamaEmbeddings(
            model=settings.ollama_embedding_model, base_url=settings.ollama_base_url
        )
        return m, f"ollama:{settings.ollama_embedding_model}"
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is not set."
            )
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.openai_embedding_model
        ), f"openai:{settings.openai_embedding_model}"
    if provider == "voyage":
        if not os.getenv("VOYAGE_API_KEY"):
            raise RuntimeError(
                "EMBEDDING_PROVIDER=voyage but VOYAGE_API_KEY is not set."
            )
        from langchain_voyageai import VoyageAIEmbeddings

        return VoyageAIEmbeddings(
            model=settings.voyage_embedding_model
        ), f"voyage:{settings.voyage_embedding_model}"
    raise RuntimeError(
        f"Unknown EMBEDDING_PROVIDER '{provider}' (use local | openai | voyage)."
    )


def get_chat_model():
    """Return (model, label, ready). ready=False -> pipeline runs in dry-run mode."""
    provider = settings.llm_provider
    try:
        if provider == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                return None, f"anthropic:{settings.anthropic_model} (no key)", False
            from langchain_anthropic import ChatAnthropic

            return (
                ChatAnthropic(model=settings.anthropic_model, temperature=0),
                f"anthropic:{settings.anthropic_model}",
                True,
            )
        if provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                return None, f"openai:{settings.openai_model} (no key)", False
            from langchain_openai import ChatOpenAI

            return (
                ChatOpenAI(model=settings.openai_model, temperature=0),
                f"openai:{settings.openai_model}",
                True,
            )
        if provider == "ollama":
            from langchain_ollama import ChatOllama

            model = ChatOllama(
                model=settings.ollama_model,
                temperature=0,
                base_url=settings.ollama_base_url,
            )
            return model, f"ollama:{settings.ollama_model}", True
    except Exception as exc:
        return None, f"{provider} (error: {exc})", False
    return None, f"unknown provider '{provider}'", False
