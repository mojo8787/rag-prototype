# Chunking Strategies

This document describes the chunking options used for document ingestion in the RAG prototype, when to use each, and how to configure them.

## Overview

Documents are split into **chunks** before embedding and storage. Chunk size and splitting strategy affect retrieval quality and LLM context usage.

## Strategies

### 1. `fixed_overlap` (default)

- **What it does**: Produces chunks of up to `chunk_size` characters with `chunk_overlap` characters overlapping between consecutive chunks. Splits preferentially at separator boundaries (paragraph → line → sentence → word) to avoid mid-word or mid-sentence cuts.
- **When to use**: General-purpose documents (reports, FAQs, mixed content). Good default when you care about retrieval recall and consistent chunk sizes.
- **Config**: `chunk_size`, `chunk_overlap`, `separators` (see below).

### 2. `by_paragraph`

- **What it does**: Splits only on paragraph boundaries (`\n\n`). Merges short paragraphs until the total length reaches `chunk_size` (max characters per chunk).
- **When to use**: When paragraph boundaries matter (contracts, articles, formal letters). Keeps one “idea” per chunk when paragraphs are well-formed.
- **Config**: `chunk_size` (used as max size per chunk).

## Configuration

| Parameter       | Env / config           | Default | Description                                      |
|----------------|------------------------|--------|--------------------------------------------------|
| `chunk_size`   | `CHUNK_SIZE`           | 512    | Max characters per chunk.                        |
| `chunk_overlap`| `CHUNK_OVERLAP`        | 64     | Overlap between consecutive chunks (fixed_overlap). |
| `separators`   | (code: `config.CHUNK_SEPARATORS`) | `["\n\n", "\n", ". ", " ", ""]` | Split order: paragraph, line, sentence, word, then character. |

Set env vars in `.env` (see `.env.example`) or pass `chunk_size` / `chunk_overlap` / `separators` into `chunk_document()` in code.

## Usage in code

```python
from src.chunking import chunk_document

# Default: fixed_overlap with config defaults
chunks = chunk_document(full_text)

# By paragraph, max 600 chars
chunks = chunk_document(full_text, strategy="by_paragraph", chunk_size=600)

# Fixed overlap with custom overlap
chunks = chunk_document(full_text, strategy="fixed_overlap", chunk_overlap=128)
```

## Adding more strategies

Chunking is implemented in `src/chunking.py`. To add a new strategy:

1. Implement a helper (e.g. `_chunk_my_strategy(text, **kwargs)`) that returns `List[str]`.
2. In `chunk_document()`, add a branch for the new strategy name and call the helper with the chosen parameters.
3. Document the strategy and its parameters in this file.
