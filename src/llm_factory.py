"""LLM and embedding factory: returns OpenAI or Azure implementations based on config."""
from __future__ import annotations

from typing import TYPE_CHECKING

from . import config

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel


def get_llm(
    model: str | None = None,
    temperature: float = 0,
) -> BaseChatModel:
    """Return LLM (OpenAI or Azure) based on LLM_PROVIDER."""
    if config.LLM_PROVIDER.lower() == "azure":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version="2024-02-15-preview",
            azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,
            model=model or config.AZURE_OPENAI_DEPLOYMENT,
            temperature=temperature,
        )
    # Default: OpenAI
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model or "gpt-4o-mini",
        temperature=temperature,
    )


def get_embeddings(
    model: str | None = None,
) -> Embeddings:
    """Return embeddings (OpenAI or Azure) based on LLM_PROVIDER."""
    if config.LLM_PROVIDER.lower() == "azure":
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version="2024-02-15-preview",
            azure_deployment=config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            model=model or config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        )
    # Default: OpenAI
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=model or "text-embedding-3-small")
