"""
config.py  --  All the knobs in one place.

The key design decision for learning: EMBEDDING runs locally and free, so you
can do steps 1-5 (the actual RAG retrieval) with zero API keys. Only the final
GENERATION step needs a model. You can use Anthropic, OpenAI, a local Ollama
model, or nothing at all (dry-run, which just shows you the prompt).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # read variables from a .env file in the project root


@dataclass
class Settings:
    # Local embedding model. Small (~90 MB), downloads once, runs on CPU.
    # Swap for "sentence-transformers/all-mpnet-base-v2" for higher quality.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Which provider to use for the GENERATE step. Options: anthropic | openai | ollama
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic").lower()

    # Model names per provider (override in .env if you like).
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


settings = Settings()


def get_chat_model():
    """
    Return (model, human_label, ready).

    'ready' is False when the chosen provider has no credentials. In that case
    the pipeline runs in dry-run mode and shows the assembled prompt instead of
    calling a model -- so the rest of the lab still works without any key.
    """
    provider = settings.llm_provider

    try:
        if provider == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                return None, f"anthropic:{settings.anthropic_model} (no key)", False
            from langchain_anthropic import ChatAnthropic
            model = ChatAnthropic(model=settings.anthropic_model, temperature=0)
            return model, f"anthropic:{settings.anthropic_model}", True

        if provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                return None, f"openai:{settings.openai_model} (no key)", False
            from langchain_openai import ChatOpenAI
            model = ChatOpenAI(model=settings.openai_model, temperature=0)
            return model, f"openai:{settings.openai_model}", True

        if provider == "ollama":
            # Ollama runs locally; no key, but the server must be running.
            from langchain_ollama import ChatOllama
            model = ChatOllama(model=settings.ollama_model, temperature=0)
            return model, f"ollama:{settings.ollama_model}", True

    except Exception as exc:  # missing package, bad config, etc.
        return None, f"{provider} (error: {exc})", False

    return None, f"unknown provider '{provider}'", False
