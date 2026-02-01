"""Document chunking strategies for RAG ingestion."""
from __future__ import annotations

from typing import List

from . import config


def _chunk_fixed_overlap(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: List[str],
) -> List[str]:
    """Split text into chunks of fixed size with overlap, trying separators first."""
    if not text or not text.strip():
        return []
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Prefer splitting on a separator
        segment = text[start:end]
        split_at = -1
        for sep in separators:
            if not sep:
                continue
            pos = segment.rfind(sep)
            if pos != -1:
                split_at = start + pos + len(sep)
                break

        if split_at > start:
            chunk = text[start:split_at].strip()
            start = split_at - chunk_overlap
            if start < 0:
                start = 0
            # Avoid re-including too much
            if chunk_overlap > 0 and start > 0:
                start = min(start, split_at - 1)
        else:
            chunk = text[start:end].strip()
            start = end - chunk_overlap

        if chunk:
            chunks.append(chunk)

    return chunks


def _chunk_by_paragraph(text: str, max_chunk_size: int) -> List[str]:
    """Split by paragraphs, merging short ones up to max_chunk_size."""
    if not text or not text.strip():
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text.strip()] if text.strip() else []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for p in paragraphs:
        p_len = len(p) + 2  # +2 for "\n\n"
        if current_len + p_len <= max_chunk_size or not current:
            current.append(p)
            current_len += p_len
        else:
            if current:
                chunks.append("\n\n".join(current))
            current = [p]
            current_len = p_len

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_document(
    text: str,
    strategy: str = "fixed_overlap",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    separators: List[str] | None = None,
    **kwargs: object,
) -> List[str]:
    """
    Chunk a document into smaller pieces for embedding and retrieval.

    Strategies:
        - fixed_overlap: Fixed-size chunks with overlap; prefers splitting on
          separators (paragraph, line, sentence, word). Use for general docs.
        - by_paragraph: Split by paragraphs, merging up to chunk_size. Use when
          paragraph boundaries matter (e.g. contracts, articles).

    Args:
        text: Raw document text.
        strategy: One of "fixed_overlap", "by_paragraph".
        chunk_size: Max characters per chunk (default from config).
        chunk_overlap: Overlap between chunks for fixed_overlap (default from config).
        separators: Split priorities for fixed_overlap (default from config).
        **kwargs: Ignored; allows future strategy options.

    Returns:
        List of chunk strings.
    """
    size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP
    seps = separators if separators is not None else config.CHUNK_SEPARATORS

    if strategy == "fixed_overlap":
        return _chunk_fixed_overlap(text, size, overlap, seps)
    if strategy == "by_paragraph":
        return _chunk_by_paragraph(text, size)
    raise ValueError(f"Unknown chunking strategy: {strategy}")
