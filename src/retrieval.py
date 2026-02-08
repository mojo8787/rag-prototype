"""Retrieval: query vector store for top-k chunks."""
from __future__ import annotations

from typing import Any, List

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from . import config


def retrieve(
    vector_store: VectorStore,
    query: str,
    top_k: int | None = None,
    **kwargs: Any,
) -> List[Document]:
    """
    Retrieve top-k chunks from the vector store for a query.

    Args:
        vector_store: Vector store from ingestion (FAISS, Chroma, etc.).
        query: Search query (e.g. user question or extraction intent).
        top_k: Number of chunks to return (default from config).
        **kwargs: Passed to vector_store.similarity_search.

    Returns:
        List of Document chunks, ordered by relevance.
    """
    k = top_k if top_k is not None else config.RETRIEVAL_TOP_K
    return vector_store.similarity_search(query, k=k, **kwargs)
