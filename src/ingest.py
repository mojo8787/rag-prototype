"""Document ingestion: load, chunk, embed, and store in vector DB."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from . import config
from .chunking import chunk_document


def _load_text_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_document(path: str | Path) -> str:
    """Load document text from a file (supports .txt and .pdf)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _load_text_file(path)
    if suffix == ".pdf":
        return _load_pdf(path)
    raise ValueError(f"Unsupported file type: {suffix}. Use .txt or .pdf.")


def ingest_documents(
    paths: List[str | Path],
    chunk_strategy: str = "fixed_overlap",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embedding_model: str = "text-embedding-3-small",
    persist_directory: str | None = None,
    collection_name: str = "rag_prototype",
) -> Chroma:
    """
    Load documents from paths, chunk them, embed, and store in Chroma.

    Returns the Chroma vector store (in-memory if persist_directory is None).
    """
    size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP

    all_chunks: List[Document] = []
    for path in paths:
        text = load_document(path)
        chunks = chunk_document(
            text,
            strategy=chunk_strategy,
            chunk_size=size,
            chunk_overlap=overlap,
        )
        base_name = Path(path).name
        for i, c in enumerate(chunks):
            all_chunks.append(
                Document(
                    page_content=c,
                    metadata={"source": base_name, "chunk_index": i},
                )
            )

    if not all_chunks:
        raise ValueError("No chunks produced from the given documents.")

    embeddings = OpenAIEmbeddings(model=embedding_model)
    chroma = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    if persist_directory:
        chroma.persist()
    return chroma
